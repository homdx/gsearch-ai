"""
skill_runner.py — исполняет скилл, описанный в YAML (см. skills/*.yaml),
вызывая примитивы из skill_primitives.py. Один и тот же раннер подходит
для любого скилла (погода, бонусы, что угодно ещё) — сам скилл решает,
что за чем идёт, раннер только исполняет шаги и прокидывает переменные.

Формат шага:
  - id: my_step            # опционально, нужен только если результат
                            # используется в следующих шагах
    action: open_url        # имя функции в skill_primitives
    <params...>              # именованные параметры этой функции,
                              # значения могут содержать {{jinja}}
    if: "{{mode}} == 'hourly'"   # опционально — шаг выполняется только
                                  # если выражение истинно
    on_error: stop | continue    # опционально, по умолчанию stop

Контекст переменных для {{...}}:
  - inputs скилла (city, mode, ...)
  - результат каждого шага с id, доступен как {{step_id.field}}
    (например {{read_forecast.result}})

Запуск:
    python3 skill_runner.py skills/weather_forecast.yaml \
        --input city=Москва --input mode=today
"""

import argparse
import configparser
import json
import sys
import time
from datetime import datetime, timezone

import yaml
from jinja2 import Template
from playwright.sync_api import sync_playwright

import skill_primitives as prim
from llm_api import LLMClient


def _log(msg: str):
    """Как prim._log() — уважает silent-режим (общий флаг prim.SILENT,
    выставленный из конфига), чтобы в silent-режиме раннер тоже не
    печатал ничего, кроме финального JSON."""
    if prim.SILENT:
        return
    print(f"[{prim._ts()}] [skill] {msg}")


def render(value, context):
    """Рекурсивно подставляет {{...}} в строки (в т.ч. внутри dict/list),
    остальные типы (bool, int, None) возвращает как есть."""
    if isinstance(value, str):
        return Template(value).render(**context)
    if isinstance(value, dict):
        return {k: render(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, context) for v in value]
    return value


def eval_condition(expr: str, context: dict) -> bool:
    """Условие вида "{{mode}} == 'hourly'" — сначала подставляем
    переменные через Jinja, потом безопасно eval'им простое сравнение
    (только литералы, без доступа к builtins)."""
    rendered = Template(expr).render(**context)
    try:
        return bool(eval(rendered, {"__builtins__": {}}, {}))
    except Exception as e:
        print(f"[skill] Не смог вычислить условие '{expr}' -> '{rendered}': {e}")
        return False


def run_skill(skill_path: str, inputs: dict, headless: bool = False,
              profile_dir: str = "chrome_profile", config_path: str = "config_vision.ini"):
    with open(skill_path, "r", encoding="utf-8") as f:
        skill = yaml.safe_load(f)

    # Проверяем обязательные inputs и подставляем default'ы —
    # без этого опечатка в --input тихо привела бы к KeyError
    # где-то в середине шагов, а не к понятной ошибке в начале.
    declared = skill.get("inputs", {})
    for name, spec in declared.items():
        if name not in inputs:
            if "default" in spec:
                inputs[name] = spec["default"]
            elif spec.get("required"):
                print(f"[skill] Ошибка: обязательный input '{name}' не задан "
                      f"(--input {name}=...)")
                sys.exit(1)

    context = dict(inputs)
    # Как в main.py: vision_client — секция [api], text_client —
    # секция [text_api], если она задана в конфиге (иначе используем
    # тот же vision_client).
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    vision_client = LLMClient.from_config(cfg)
    if cfg.has_section("text_api"):
        text_client = LLMClient.from_config(cfg, config_section="text_api")
    else:
        text_client = vision_client

    # Silent-режим — как RUN_STATS["silent"] в main.py: читается из
    # конфига (секция [api], ключ silent), никакого текстового лога в
    # консоль, единственный вывод — финальный JSON в конце. Полезно,
    # когда результат скилла нужно скормить другому процессу/скрипту.
    prim.SILENT = cfg.getboolean("api", "silent", fallback=False)

    started_at = datetime.now(timezone.utc)
    t_run_start = time.monotonic()
    error = None
    success = False
    final_text = None

    _log(f"Запускаю скилл '{skill.get('name')}' с inputs={inputs}")

    step_error = None  # (message,) - если шаг провалился с on_error=stop

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            profile_dir, headless=headless, viewport={"width": 1024, "height": 768},
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        for step in skill["steps"]:
            action_name = step["action"]
            step_id = step.get("id")

            # ВАЖНО: сначала проверяем "if" на СЫРОМ шаге (только само
            # условие рендерится через Jinja), и только если условие
            # истинно — рендерим остальные параметры шага. Раньше был
            # баг: render(step, context) рендерил ВЕСЬ шаг, включая
            # параметры вроде {{shot.path}}, ДО проверки if — если шаг
            # 'shot' был пропущен (его не было в context), рендер
            # следующего шага 'read_forecast' падал с UndefinedError,
            # даже если read_forecast тоже должен был быть пропущен.
            if "if" in step and not eval_condition(step["if"], context):
                _log(f"Пропускаю шаг '{step_id or action_name}' (условие ложно)")
                continue

            step = render(step, context)

            fn = getattr(prim, action_name, None)
            if not fn:
                step_error = f"Неизвестный примитив: '{action_name}'"
                _log(step_error)
                break

            params = {k: v for k, v in step.items()
                      if k not in ("id", "action", "if", "on_error")}

            # Автоматически подставляем page/client, если функция их
            # ожидает — так в YAML не нужно про них ничего писать.
            import inspect
            sig = inspect.signature(fn)
            call_kwargs = dict(params)
            if "page" in sig.parameters:
                call_kwargs["page"] = page
            if "client" in sig.parameters:
                call_kwargs["client"] = vision_client
            if "text_client" in sig.parameters:
                call_kwargs["text_client"] = text_client

            _log(f"-> {step_id or action_name}: {action_name}({params})")
            result = fn(**call_kwargs)
            _log(f"<- {result if not isinstance(result, dict) or len(str(result)) < 300 else str(result)[:300] + '...'}")

            if isinstance(result, dict) and not result.get("ok", True):
                if step.get("on_error", "stop") == "stop":
                    step_error = (f"Шаг '{step_id or action_name}' провалился: "
                                  f"{result.get('reason')}")
                    _log(step_error + ". Останавливаюсь.")
                    if step_id:
                        context[step_id] = result
                    break
                else:
                    _log("Шаг провалился, но on_error=continue, иду дальше")

            if step_id:
                context[step_id] = result

        browser.close()

    output_tmpl = skill.get("output")
    if step_error is None and output_tmpl:
        try:
            final_text = render(output_tmpl, context)
            success = True
        except Exception as e:
            step_error = f"Не смог собрать output: {e}"
    error = step_error

    # Та же структура и то же поведение, что write_result_and_exit() в
    # main.py: всегда собираем JSON, в silent-режиме печатаем ТОЛЬКО
    # его и ничего не пишем на диск, иначе сохраняем в skill_result.json
    # (рядом с run_skill) И печатаем тот же JSON в консоль.
    finished_at = datetime.now(timezone.utc)
    duration_sec = round(time.monotonic() - t_run_start, 1)
    result = {
        "skill": skill.get("name"),
        "inputs": inputs,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_sec,
        "llm_requests_total": prim.RUN_STATS["llm_requests_total"],
        "success": success,
        "final_answer": final_text,
        "error": error,
    }
    result_json = json.dumps(result, ensure_ascii=False, indent=2)

    if prim.SILENT:
        print(result_json)
    else:
        result_path = "skill_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        _log(f"Результат сохранён в {result_path}")
        print(result_json)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Исполнить скилл из YAML")
    parser.add_argument("skill_file")
    parser.add_argument("--input", action="append", default=[],
                         help="key=value, можно несколько раз")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--profile-dir", default="chrome_profile")
    parser.add_argument("--config", default="config_vision.ini")
    args = parser.parse_args()

    inputs = {}
    for item in args.input:
        k, _, v = item.partition("=")
        inputs[k] = v

    run_skill(args.skill_file, inputs, headless=args.headless,
              profile_dir=args.profile_dir, config_path=args.config)
