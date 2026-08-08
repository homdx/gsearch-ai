"""
skill_primitives.py — низкоуровневые "глаголы" (примитивы), вынесенные
из main.py, чтобы ими мог пользоваться универсальный skill_runner.py
(см. skills/*.yaml) без хардкода конкретного флоу (поиск бонусов).

main.py НЕ ИЗМЕНЁН и продолжает работать как раньше — это отдельный
слой поверх тех же идей (goto с ретраями, увеличенный скриншот зоны,
vision-анализ), но переиспользуемый для ЛЮБОГО скилла (погода,
бонусы, что угодно), который описывается в YAML.

Каждый примитив:
  - принимает page (Playwright) и/или client (LLMClient) первым(и)
    аргументом(ами), плюс параметры из шага скилла;
  - возвращает dict с результатом, который runner кладёт в контекст
    под id шага (доступно дальше как {{step_id.field}});
  - никогда не бросает исключение наружу без необходимости — при
    ожидаемых сбоях (элемент не найден и т.п.) возвращает
    {"ok": False, "reason": "..."}, чтобы runner мог решить, что
    делать (retry / fallback / стоп), а не падал всем процессом.
"""

import os
import time

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from llm_api import LLMClient

# ВАЖНО: убрал самопальную SKILL_DEBUG - в проекте такой переменной
# никогда не было (только LLM_DEBUG внутри llm_api.py для сырых
# запросов/ответов модели). Подробный лог DOM-текста внутри
# extract_text() теперь включается тем же способом, которым в проекте
# уже включается любая отладка - флагом debug у самого LLMClient
# (config_vision.ini [api]/[text_api] debug=true, либо LLM_DEBUG=1),
# а не отдельной новой переменной окружения.

# SILENT управляется из skill_runner.py (читается из config_vision.ini,
# как RUN_STATS["silent"] в main.py) - когда True, _log() ничего не
# печатает.
SILENT = False

# Та же идея, что RUN_STATS в main.py - копится сюда, наружу читается
# из skill_runner.py при сборке итогового result.json.
RUN_STATS = {"llm_requests_total": 0}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# Те же 9 зон, что и в main.py (capture_zoomed_region / ZOOM_REGIONS) -
# держим отдельную копию здесь, чтобы skill_primitives не тянул за
# собой весь main.py (там 2000+ строк логики конкретного флоу бонусов,
# импортировать его целиком ради одной таблицы координат избыточно и
# создаёт риск побочных эффектов уровня модуля).
ZOOM_REGIONS = {
    "top_left":       (0.0, 0.0, 0.5, 0.33),
    "top_center":     (0.25, 0.0, 0.75, 0.33),
    "top_right":      (0.5, 0.0, 1.0, 0.33),
    "middle_left":    (0.0, 0.33, 0.5, 0.66),
    "middle_center":  (0.25, 0.33, 0.75, 0.66),
    "middle_right":   (0.5, 0.33, 1.0, 0.66),
    "bottom_left":    (0.0, 0.66, 0.5, 1.0),
    "bottom_center":  (0.25, 0.66, 0.75, 1.0),
    "bottom_right":   (0.5, 0.66, 1.0, 1.0),
}

GOTO_RETRIES = 3
GOTO_RETRY_WAIT_SEC = 5


def _log(msg: str):
    """Таймштамп [ЧЧ:ММ:СС] - та же идея, что log() в main.py: без
    этого непонятно, скрипт завис или локальная LLM просто долго
    думает (может идти минуты). В silent-режиме молчит - как
    RUN_STATS["silent"] в main.py."""
    if SILENT:
        return
    print(f"[{_ts()}] [skill] {msg}")


# ---------------------------------------------------------------------
# Навигация
# ---------------------------------------------------------------------

def open_url(page, url: str, timeout: int = 20000) -> dict:
    """Аналог goto_with_retry из main.py: не падает на кратковременных
    сетевых сбоях, ждёт domcontentloaded (надёжнее чем дефолтный load),
    плюс небольшая доп. пауза для SPA-сайтов, дорисовывающих контент
    через JS уже после domcontentloaded."""
    last_exc = None
    for attempt in range(1, GOTO_RETRIES + 1):
        try:
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            break
        except Exception as e:
            last_exc = e
            if attempt < GOTO_RETRIES:
                _log(f"Не удалось открыть {url} (попытка {attempt}/{GOTO_RETRIES}): {e}")
                time.sleep(GOTO_RETRY_WAIT_SEC)
            else:
                return {"ok": False, "reason": str(e)}
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    time.sleep(1.5)
    return {"ok": True, "url": page.url}


def wait(page, seconds: float = 1.0) -> dict:
    time.sleep(seconds)
    return {"ok": True}


# ---------------------------------------------------------------------
# Взаимодействие с элементами: селектор -> fallback на vision
# ---------------------------------------------------------------------

def click(page, client: LLMClient = None, selector: str = None,
          description: str = None, screenshot_dir: str = "screens",
          timeout: int = 3000) -> dict:
    """Комбо-стратегия, как договорились: сначала пробуем обычный CSS
    селектор (быстро, дёшево, детерминированно). Если он не задан или
    элемент не найден за timeout — просим vision-модель определить
    ПРИМЕРНУЮ зону экрана (та же идея, что capture_zoomed_region /
    ZOOM_REGIONS в main.py) по текстовому описанию цели и кликаем в
    центр этой зоны координатами мыши."""
    if selector:
        try:
            page.click(selector, timeout=timeout)
            return {"ok": True, "method": "selector", "selector": selector}
        except PlaywrightTimeoutError:
            _log(f"Селектор '{selector}' не найден за {timeout}мс, fallback на vision")
        except Exception as e:
            _log(f"Клик по селектору '{selector}' упал: {e}, fallback на vision")

    if not description:
        return {"ok": False, "reason": "элемент не найден по селектору, "
                                        "а description для vision-fallback не задан"}
    if not client:
        return {"ok": False, "reason": "нужен client (LLMClient) для vision-fallback"}

    os.makedirs(screenshot_dir, exist_ok=True)
    shot_path = os.path.join(screenshot_dir, "click_lookup.png")
    page.screenshot(path=shot_path)

    prompt = (
        f"На скриншоте страницы найди элемент: \"{description}\". "
        f"В какой из 9 зон экрана он находится? Зоны: "
        f"top_left, top_center, top_right, middle_left, middle_center, "
        f"middle_right, bottom_left, bottom_center, bottom_right. "
        f"Ответь ОДНИМ словом — названием зоны."
    )
    answer = client.chat_vision(prompt, shot_path).strip().lower()
    zone = next((z for z in ZOOM_REGIONS if z in answer), None)
    if not zone:
        return {"ok": False, "reason": f"vision не смог определить зону: {answer!r}"}

    x0, y0, x1, y1 = ZOOM_REGIONS[zone]
    viewport = page.viewport_size
    cx = viewport["width"] * (x0 + x1) / 2
    cy = viewport["height"] * (y0 + y1) / 2
    page.mouse.click(cx, cy)
    return {"ok": True, "method": "vision", "zone": zone}


def type_text(page, value: str, selector: str = None, press_enter: bool = False) -> dict:
    """Печатает в текущий фокус (если selector не задан — предполагаем,
    что фокус уже стоит на нужном поле после предыдущего click) или в
    явно указанный selector."""
    try:
        if selector:
            page.fill(selector, value)
        else:
            page.keyboard.type(value)
        if press_enter:
            page.keyboard.press("Enter")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ---------------------------------------------------------------------
# Чтение страницы
# ---------------------------------------------------------------------

def screenshot(page, save_path: str = "screens/step.png", zoom_region: str = None) -> dict:
    """Полный скриншот или увеличенный клип одной из 9 зон (та же идея,
    что capture_zoomed_region в main.py — полезно, когда нужный текст
    мелкий, например ячейка почасового прогноза)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    try:
        if zoom_region and zoom_region in ZOOM_REGIONS:
            x0, y0, x1, y1 = ZOOM_REGIONS[zoom_region]
            vp = page.viewport_size
            clip = {
                "x": vp["width"] * x0, "y": vp["height"] * y0,
                "width": vp["width"] * (x1 - x0), "height": vp["height"] * (y1 - y0),
            }
            page.screenshot(path=save_path, clip=clip)
        else:
            page.screenshot(path=save_path)
        return {"ok": True, "path": save_path}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def ask_vision(client: LLMClient, prompt: str, image_path: str) -> dict:
    via_proxy = client.proxy_enabled and not client._is_local_host
    _log(f"-> vision-запрос в LLM для {image_path} (модель: {client.model}, "
         f"{'через прокси' if via_proxy else 'напрямую'})...")
    t0 = time.monotonic()
    try:
        answer = client.chat_vision(prompt, image_path)
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    dt = time.monotonic() - t0
    RUN_STATS["llm_requests_total"] += 1
    _log(f"<- ответ получен за {dt:.1f} сек")
    return {"ok": True, "result": answer, "duration_sec": round(dt, 1)}


def ask_text(text_client: LLMClient, prompt: str, system: str = "Ты помощник, отвечай кратко и по делу.") -> dict:
    via_proxy = text_client.proxy_enabled and not text_client._is_local_host
    _log(f"-> текстовый запрос в LLM (модель: {text_client.model}, "
         f"{'через прокси' if via_proxy else 'напрямую'})...")
    t0 = time.monotonic()
    try:
        answer = text_client.chat(system, prompt)
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    dt = time.monotonic() - t0
    RUN_STATS["llm_requests_total"] += 1
    _log(f"<- ответ получен за {dt:.1f} сек")
    return {"ok": True, "result": answer, "duration_sec": round(dt, 1)}


def extract_text(page, text_client: LLMClient, question: str,
                  not_found_marker: str = "NOT_FOUND", max_chars: int = 20000,
                  visible_only: bool = False) -> dict:
    """Text-first чтение страницы — то же самое, ради чего в main.py
    есть весь текстовый путь (find_bonus_blocks/try_full_page_text)
    ДО какого-либо vision: HTML/DOM текст читается моделью напрямую,
    без промежуточного "перевода в картинку", поэтому числа (температура,
    проценты, цены) не искажаются, запрос дешевле и быстрее, чем
    vision-запрос со скриншотом.

    По умолчанию (visible_only=False) берём ВЕСЬ HTML через
    page.content() и парсим текст через BeautifulSoup (как в main.py:
    extract_main_page_text/find_bonus_blocks) — а не только видимую
    часть через page.inner_text(). Это важно: сайты с вкладками
    (например "Сегодня"/"Завтра"/"На 10 дней" у Яндекс.Погоды) обычно
    рендерят контент ВСЕХ вкладок в DOM заранее и просто прячут
    неактивные через display:none/hidden — inner_text() такой текст
    НЕ возвращает (Playwright уважает CSS-видимость), из-за чего
    реальный случай: "Завтра" был в HTML, но extract_text его не
    находил. page.content() + BeautifulSoup видит весь HTML независимо
    от того, что сейчас показано на экране.

    visible_only=True — старое поведение (page.inner_text), пригодится
    для страниц, где нужно ИМЕННО то, что видно (например динамически
    подгружаемый контент, которого ещё нет в HTML при первой загрузке)."""
    try:
        if visible_only:
            raw_text = page.inner_text("body")
        else:
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            raw_text = soup.get_text(separator="\n")
            # Схлопываем пустые строки - HTML обычно даёт кучу
            # пустых строк из-за форматирования разметки.
            raw_text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
    except Exception as e:
        return {"ok": False, "reason": f"не смог прочитать текст страницы: {e}"}

    raw_text = raw_text.strip()[:max_chars]
    if not raw_text:
        return {"ok": True, "result": not_found_marker, "sufficient": False}
    if text_client.debug:
        _log(f"[extract_text] DOM-текст ({len(raw_text)} симв., первые 300):\n{raw_text[:300]}")

    prompt = (
        f"Вот текст страницы (может содержать лишний мусор — меню, "
        f"рекламу и т.п., игнорируй его):\n\n{raw_text}\n\n"
        f"Вопрос: {question}\n\n"
        f"Если ответ ЕСТЬ в тексте выше — ответь на вопрос кратко и "
        f"по существу. Если конкретных данных для ответа в тексте НЕТ "
        f"(например, значения отрисованы графикой, а не текстом) — "
        f"ответь ровно одним словом: {not_found_marker}"
    )
    via_proxy = text_client.proxy_enabled and not text_client._is_local_host
    _log(f"-> extract_text: текстовый запрос в LLM (модель: {text_client.model}, "
         f"{'через прокси' if via_proxy else 'напрямую'})...")
    t0 = time.monotonic()
    try:
        answer = text_client.chat(
            "Ты внимательно читаешь текст страницы и отвечаешь на вопрос "
            "только на основе этого текста, не выдумывая данные.",
            prompt,
        ).strip()
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    dt = time.monotonic() - t0
    RUN_STATS["llm_requests_total"] += 1
    _log(f"<- ответ получен за {dt:.1f} сек: {answer[:200]!r}")

    sufficient = not_found_marker not in answer.upper()
    return {"ok": True, "result": answer, "sufficient": sufficient, "duration_sec": round(dt, 1)}
