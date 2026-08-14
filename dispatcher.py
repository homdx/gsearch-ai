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
import time
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_api
from llm_api import LLMClient

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
UNKNOWN_INTENT = "unknown"

# Silent-режим — как RUN_STATS["silent"] в main.py и prim.SILENT в
# skill_runner.py/skill_primitives.py: читается из конфига (секция [api],
# ключ silent). Раньше dispatcher.py его вообще не проверял и всегда
# печатал свои служебные строки ("[dispatcher] ...") напрямую через
# print(), из-за чего silent-режим ломался именно на уровне диспетчера -
# skill_runner ниже по стеку молчал правильно, а сам dispatcher - нет.
SILENT = False


def dispatcher_print(*args, **kwargs):
    """Замена print() для служебного вывода dispatcher.py - в
    silent-режиме молчит, как log()/silent_aware_print() в main.py."""
    if SILENT:
        return
    print(*args, **kwargs)


def _token_usage_delta(before: dict, after: dict):
    """Разница между двумя снимками llm_api.TOKEN_STATS. Возвращает
    None, если за это время usage ни разу не пришёл от провайдера
    (requests_with_usage не увеличился) - т.е. как и в get_token_stats_
    or_none(), не показываем "0 токенов", если реально ничего не мерили."""
    if after["requests_with_usage"] == before["requests_with_usage"]:
        return None
    return {
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "requests_with_usage": after["requests_with_usage"] - before["requests_with_usage"],
    }


def _merge_token_usage(a, b):
    """Складывает два token_usage-словаря (каждый может быть None, если
    usage для этой части не мерился). None + None -> None."""
    if a is None and b is None:
        return None
    a = a or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests_with_usage": 0}
    b = b or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests_with_usage": 0}
    return {
        "prompt_tokens": a["prompt_tokens"] + b["prompt_tokens"],
        "completion_tokens": a["completion_tokens"] + b["completion_tokens"],
        "total_tokens": a["total_tokens"] + b["total_tokens"],
        "requests_with_usage": a["requests_with_usage"] + b["requests_with_usage"],
    }


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
        dispatcher_print(f"[dispatcher] Не смог распарсить JSON от LLM: {e}\nСырой ответ:\n{raw}")
        return {"intent": UNKNOWN_INTENT, "inputs": {}, "confidence": 0.0}


# ─── Запуск скилла ────────────────────────────────────────────────────────────

def run_skill(skill_yaml: str, inputs: dict, config_path: str,
              profile_dir: str = None) -> dict:
    input_args = []
    for k, v in inputs.items():
        input_args += ["--input", f"{k}={v}"]

    # БАГ (был): "skill_runner.py" - относительный путь, т.е. запуск
    # dispatcher.py не из папки проекта падал с "can't open file".
    # Берём соседний файл рядом с самим dispatcher.py.
    runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "skill_runner.py")

    cmd = [
        sys.executable, runner_path, skill_yaml,
        *input_args,
        "--config", config_path,
    ]

    # БАГ (был): profile_dir не прокидывался вообще - если ручной логин
    # через step1_open_chrome.py делался в НЕ дефолтный профиль
    # (--profile-dir chrome_profile2), скилл всё равно уходил в дефолтный,
    # незалогиненный, и получал капчу от Google. Профиль задаётся
    # переменной окружения CHROME_PROFILE_DIR (отдельного CLI-флага нет:
    # у dispatcher.py весь sys.argv - это текст запроса пользователя).
    if profile_dir:
        cmd += ["--profile-dir", profile_dir]
    dispatcher_print(f"[dispatcher] → {' '.join(cmd)}")

    # Popen со стримингом: stdout читаем построчно и сразу печатаем —
    # LLM_DEBUG, логи skill_runner и прогресс видны в реальном времени,
    # а не после завершения всего прогона.
    # stderr=None — наследуем у родителя (идёт в терминал напрямую).
    # Все строки stdout собираем, чтобы в конце распарсить итоговый JSON.
    # В silent-режиме сам skill_runner.py (через prim.SILENT) печатает в
    # свой stdout ТОЛЬКО финальный JSON - поэтому стриминг здесь можно не
    # отключать отдельно, но дублировать его dispatcher_print'ом не нужно:
    # используем silent-aware вывод, чтобы соблюсти контракт "ничего кроме
    # финального ответа" даже если skill_runner всё же что-то напечатает.
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
        if not SILENT:
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

def dispatch(user_input: str, client: LLMClient, skills: dict, config_path: str,
             profile_dir: str = None) -> dict:
    """Возвращает структурированный результат в ТОМ ЖЕ формате, что и
    write_result_and_exit() в main.py / финальный JSON в skill_runner.py:
    started_at/finished_at/duration_seconds/llm_requests_total — чтобы
    и через dispatcher.py, и напрямую было видно время выполнения и
    количество LLM-запросов, а не только сам ответ.

    llm_requests_total здесь = 1 запрос классификации intent (делает сам
    dispatcher, через свой client) + llm_requests_total из JSON, который
    вернул skill_runner.py (запросы внутри самого скилла) - т.е. это
    ПОЛНАЯ сумма LLM-запросов за весь вызов, а не только часть скилла.
    """
    started_at = datetime.now(timezone.utc)
    t_start = time.monotonic()

    dispatcher_print(f"\n[dispatcher] Вход: «{user_input}»")

    # Снимок TOKEN_STATS ДО классификации - чтобы посчитать usage именно
    # этого вызова classify(), а не накопленную сумму за весь процесс
    # (важно для REPL-режима, где dispatch() вызывается много раз подряд
    # в одном и том же процессе, и llm_api.TOKEN_STATS копится сквозным
    # итогом, если его не снимать разницей на каждом вызове).
    tokens_before = dict(llm_api.TOKEN_STATS)

    result = classify(user_input, client, skills)
    intent = result.get("intent", UNKNOWN_INTENT)
    inputs = result.get("inputs", {})
    confidence = result.get("confidence", 0.0)
    classify_llm_requests = 1  # один вызов client.chat() внутри classify()

    dispatcher_print(f"[dispatcher] Намерение: {intent} (confidence={confidence:.2f}), inputs={inputs}")

    classify_token_usage = _token_usage_delta(tokens_before, llm_api.TOKEN_STATS)

    def finalize(success: bool, error, final_answer, skill_meta: dict = None) -> dict:
        finished_at = datetime.now(timezone.utc)
        duration_sec = round(time.monotonic() - t_start, 1)
        llm_requests_total = classify_llm_requests + (
            (skill_meta or {}).get("llm_requests_total", 0)
        )
        token_usage = _merge_token_usage(
            classify_token_usage, (skill_meta or {}).get("token_usage")
        )
        return {
            "input": user_input,
            "intent": intent,
            "inputs": inputs,
            "confidence": confidence,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_sec,
            "llm_requests_total": llm_requests_total,
            "token_usage": token_usage,
            "success": success,
            "final_answer": final_answer,
            "error": error,
        }

    if intent == UNKNOWN_INTENT or intent not in skills:
        return finalize(False, f"Не понял запрос или нет подходящего скилла для: «{user_input}»", None)

    if confidence < 0.5:
        return finalize(False, f"Низкая уверенность ({confidence:.2f}) в распознавании: «{user_input}»", None)

    skill_result = run_skill(skills[intent]["yaml"], inputs, config_path, profile_dir)

    if skill_result.get("success"):
        return finalize(True, None, skill_result.get("final_answer") or "(пустой ответ)", skill_result)
    else:
        return finalize(False, f"Скилл завершился с ошибкой: {skill_result.get('error')}", None, skill_result)


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

    # silent читается ИЗ КОНФИГА ДО первого вывода — как RUN_STATS["silent"]
    # в main.py и prim.SILENT в skill_runner.py — иначе строки вида
    # "[dispatcher] Клиент классификации: ..." успели бы напечататься до
    # того, как мы узнали, что должны молчать. Именно в этом была причина
    # бага: SILENT никогда не выставлялся, поэтому dispatcher.py печатал
    # свои служебные сообщения независимо от silent = true/false в конфиге.
    global SILENT
    SILENT = cfg.getboolean("api", "silent", fallback=False)

    # Используем text_api (как skill_runner/main для текста), а не [api] (vision).
    # Если text_api не задан — fallback на [api], как в skill_runner.py.
    if cfg.has_section("text_api"):
        section = "text_api"
    else:
        section = "api"

    active = cfg.get(section, "active", fallback="local")
    model  = cfg.get(f"{section}_{active}", "model", fallback="?")
    base   = cfg.get(f"{section}_{active}", "base_url", fallback="?")
    dispatcher_print(f"[dispatcher] Клиент классификации: секция=[{section}] active={active!r} "
                      f"model={model!r} base_url={base!r}")

    return LLMClient.from_config(cfg, config_section=section), cfg


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_vision.ini"
    )
    client, cfg = load_client(config_path)

    # Профиль Chrome: если не задан - skill_runner.py возьмёт свой
    # дефолт (SCRIPT_DIR/chrome_profile), тот же, что у
    # step1_open_chrome.py. Переопределить можно так:
    #     CHROME_PROFILE_DIR=/путь/chrome_profile2 python3 dispatcher.py "запрос"
    profile_dir = os.environ.get("CHROME_PROFILE_DIR") or None
    if profile_dir:
        profile_dir = os.path.abspath(os.path.expanduser(profile_dir))

    # Загружаем скиллы из YAML при старте
    skills = load_skills()
    if not skills:
        # Отсутствие скиллов — фатальная ошибка запуска, а не обычный лог;
        # печатаем её даже в silent-режиме, иначе процесс молча падает.
        print(f"[dispatcher] Не найдено ни одного скилла в {SKILLS_DIR}")
        sys.exit(1)
    dispatcher_print(f"[dispatcher] Загружено скиллов: {list(skills.keys())}")

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        result = dispatch(user_input, client, skills, config_path, profile_dir)
        if SILENT:
            # silent = true -> ТОЛЬКО JSON на stdout, как в main.py и
            # skill_runner.py - никакого текста ни до, ни после.
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            answer = result.get("final_answer") if result.get("success") else result.get("error")
            print(f"\n{'='*60}\n{answer}\n{'='*60}")
    else:
        dispatcher_print("Диспетчер запущен. Введите запрос (или 'выход'):")
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                dispatcher_print("\nВыход.")
                break
            if not user_input or user_input.lower() in ("выход", "exit", "quit", "q"):
                break
            result = dispatch(user_input, client, skills, config_path, profile_dir)
            if SILENT:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                answer = result.get("final_answer") if result.get("success") else result.get("error")
                print(f"\n{'='*60}\n{answer}\n{'='*60}")


if __name__ == "__main__":
    main()
