"""
dispatcher.py — принимает свободный текст, классифицирует через LLM
и запускает нужный скилл. Всё — имена параметров, допустимые значения,
подсказки, примеры фраз — читается ДИНАМИЧЕСКИ из YAML-файлов скиллов.
В dispatcher.py ничего не захардкожено про конкретные скиллы.

Запуск:
    python3 dispatcher.py "Погода Казань"
    python3 dispatcher.py "прогноз на 3 дня Новосибирск"
    python3 dispatcher.py          # интерактивный REPL
"""

import configparser
import glob
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_api import LLMClient

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
UNKNOWN_INTENT = "unknown"


# ─── Загрузка реестра скиллов прямо из YAML ──────────────────────────────────

def load_skills(skills_dir: str = SKILLS_DIR) -> dict:
    """
    Сканирует skills/*.yaml и строит реестр.
    skill_runner.py игнорирует неизвестные ключи верхнего уровня,
    поэтому dispatcher_hints и расширенные inputs.*.values/hint
    можно хранить прямо в том же файле без правок раннера.
    """
    skills = {}
    for path in sorted(glob.glob(os.path.join(skills_dir, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
        skills[name] = {
            "yaml": path,
            "description": (data.get("description") or "").strip(),
            "examples": (data.get("dispatcher_hints") or {}).get("examples", []),
            "inputs": data.get("inputs") or {},
        }
    return skills


# ─── Построение промпта для LLM из метаданных скиллов ────────────────────────

def _format_input_spec(param: str, spec: dict) -> str:
    """Описание одного параметра для промпта классификатора."""
    parts = [f'  - "{param}"']
    if spec.get("required"):
        parts[0] += " (обязательный)"
    if spec.get("default") is not None:
        parts[0] += f' (default: {spec["default"]!r})'
    if spec.get("hint"):
        parts.append(f'    Подсказка: {spec["hint"].strip()}')
    if spec.get("values"):
        vals = spec["values"]
        # values — dict {value: "когда использовать"} или list
        if isinstance(vals, dict):
            formatted = "; ".join(f'"{k}" — {v}' for k, v in vals.items())
        else:
            formatted = " | ".join(str(v) for v in vals)
        parts.append(f"    Допустимые значения: {formatted}")
    return "\n".join(parts)


def build_classify_prompt(skills: dict) -> str:
    """
    Формирует системный промпт для LLM-классификатора целиком
    из метаданных YAML. Добавь новый скилл — промпт обновится сам.
    """
    skill_blocks = []
    for name, info in skills.items():
        block = [f'Скилл "{name}":']
        block.append(f'  Описание: {info["description"]}')
        if info["examples"]:
            block.append(f'  Примеры запросов: {", ".join(repr(e) for e in info["examples"][:4])}')
        if info["inputs"]:
            block.append("  Параметры (inputs):")
            for param, spec in info["inputs"].items():
                block.append(_format_input_spec(param, spec or {}))
        skill_blocks.append("\n".join(block))

    skills_text = "\n\n".join(skill_blocks)

    return f"""Ты диспетчер запросов. Получаешь фразу пользователя и возвращаешь ТОЛЬКО JSON, без пояснений, без ```json``` блоков.

Доступные скиллы:

{skills_text}

Формат ответа:
{{
  "intent": "<skill_name> или \\"{UNKNOWN_INTENT}\\"",
  "inputs": {{
    "param1": "value1",
    ...
  }},
  "confidence": 0.0-1.0
}}

Правила:
- Заполняй inputs строго из описания параметров скилла выше.
- Используй точные значения из "Допустимые значения", не придумывай свои.
- Если запрос не соответствует ни одному скиллу — intent = "{UNKNOWN_INTENT}".
- confidence: 0.9+ если явно понятно, 0.6-0.9 если вероятно, <0.6 если неясно.
"""


# ─── Классификация ────────────────────────────────────────────────────────────

def classify(user_input: str, client: LLMClient, skills: dict) -> dict:
    system = build_classify_prompt(skills)
    raw = client.chat(system, f'Запрос пользователя: "{user_input}"').strip()

    # Снимаем ```json если модель всё равно добавила
    if "```" in raw:
        raw = raw.split("```", 1)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print(f"[dispatcher] Не смог распарсить JSON от LLM: {e}\nСырой ответ:\n{raw}")
        return {"intent": UNKNOWN_INTENT, "inputs": {}, "confidence": 0.0}


# ─── Запуск скилла ────────────────────────────────────────────────────────────

def run_skill(skill_yaml: str, inputs: dict, config_path: str) -> dict:
    input_args = []
    for k, v in inputs.items():
        input_args += ["--input", f"{k}={v}"]

    cmd = [
        sys.executable, "skill_runner.py", skill_yaml,
        *input_args,
        "--config", config_path,
    ]
    print(f"[dispatcher] → {' '.join(cmd)}")

    # Popen со стримингом: stdout читаем построчно и сразу печатаем —
    # LLM_DEBUG, логи skill_runner и прогресс видны в реальном времени,
    # а не после завершения всего прогона.
    # stderr=None — наследуем у родителя (идёт в терминал напрямую).
    # Все строки stdout собираем, чтобы в конце распарсить итоговый JSON.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except Exception as e:
        return {"success": False, "error": str(e), "final_answer": None}

    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)   # стриминг в терминал
        lines.append(line)
    proc.wait()

    stdout = "".join(lines).strip()
    if not stdout:
        return {"success": False, "error": "пустой stdout от skill_runner", "final_answer": None}

    # skill_runner печатает итоговый JSON как последний блок в stdout.
    # JSON многострочный — ищем последнее вхождение "\n{" или "^{" и
    # парсим от него до конца. Нельзя парсить построчно: одиночная строка
    # "{"  — это незаконченный JSON, а не весь блок.
    last_brace = stdout.rfind("\n{")
    if last_brace >= 0:
        candidate = stdout[last_brace:].strip()
    else:
        # JSON с самого начала stdout (без предшествующих логов)
        candidate = stdout

    try:
        return json.loads(candidate)
    except Exception:
        return {"success": False, "error": f"не нашёл JSON в stdout:\n{stdout[-500:]}", "final_answer": None}


# ─── Основной цикл ────────────────────────────────────────────────────────────

def dispatch(user_input: str, client: LLMClient, skills: dict, config_path: str) -> str:
    print(f"\n[dispatcher] Вход: «{user_input}»")

    result = classify(user_input, client, skills)
    intent = result.get("intent", UNKNOWN_INTENT)
    inputs = result.get("inputs", {})
    confidence = result.get("confidence", 0.0)

    print(f"[dispatcher] Намерение: {intent} (confidence={confidence:.2f}), inputs={inputs}")

    if intent == UNKNOWN_INTENT or intent not in skills:
        return f"Не понял запрос или нет подходящего скилла для: «{user_input}»"

    if confidence < 0.5:
        return f"Низкая уверенность ({confidence:.2f}) в распознавании: «{user_input}»"

    skill_result = run_skill(skills[intent]["yaml"], inputs, config_path)

    if skill_result.get("success"):
        return skill_result.get("final_answer") or "(пустой ответ)"
    else:
        return f"Скилл завершился с ошибкой: {skill_result.get('error')}"


def load_client(config_path: str) -> LLMClient:
    """
    Загружает клиент для классификации.
    Приоритет: [text_api] (remote) → [api] (fallback).
    Такой же порядок, как в skill_runner.py и main.py для текстовых запросов.
    Явно логирует какой провайдер/модель выбраны — чтобы не гадать.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"[dispatcher] Конфиг не найден: {config_path}\n"
            f"  Текущий каталог: {os.getcwd()}\n"
            f"  Укажите правильный путь или запустите из папки проекта."
        )

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    # Используем text_api (как skill_runner/main для текста), а не [api] (vision).
    # Если text_api не задан — fallback на [api], как в skill_runner.py.
    if cfg.has_section("text_api"):
        section = "text_api"
    else:
        section = "api"

    active = cfg.get(section, "active", fallback="local")
    model  = cfg.get(f"{section}_{active}", "model", fallback="?")
    base   = cfg.get(f"{section}_{active}", "base_url", fallback="?")
    print(f"[dispatcher] Клиент классификации: секция=[{section}] active={active!r} "
          f"model={model!r} base_url={base!r}")

    return LLMClient.from_config(cfg, config_section=section), cfg


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_vision.ini"
    )
    client, cfg = load_client(config_path)

    # Загружаем скиллы из YAML при старте
    skills = load_skills()
    if not skills:
        print(f"[dispatcher] Не найдено ни одного скилла в {SKILLS_DIR}")
        sys.exit(1)
    print(f"[dispatcher] Загружено скиллов: {list(skills.keys())}")

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        answer = dispatch(user_input, client, skills, config_path)
        print(f"\n{'='*60}\n{answer}\n{'='*60}")
    else:
        print("Диспетчер запущен. Введите запрос (или 'выход'):")
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                break
            if not user_input or user_input.lower() in ("выход", "exit", "quit", "q"):
                break
            answer = dispatch(user_input, client, skills, config_path)
            print(f"\n{'='*60}\n{answer}\n{'='*60}")


if __name__ == "__main__":
    main()
