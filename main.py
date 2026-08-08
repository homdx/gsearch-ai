"""
main.py — точка входа: разбор аргументов командной строки, подготовка
окружения (профиль Chrome, конфиг) и запуск пайплайна поиска раздела
бонусов на сайте.

Вся бизнес-логика (константы, состояние прогона RUN_STATS, работа с
Playwright/LLM, извлечение и анализ текста, скриншоты, синтез ответа)
вынесена в pipeline_core.py — здесь только "дирижёр": argparse, main()
и запись итогового result.json.

1. Открываем URL через Playwright (используем сохранённый профиль Chrome
   из step1_open_chrome.py — уже залогинен, без капч).
2. Скачиваем HTML страницы.
3. Ищем в HTML блок(и), которые похожи на "бонусы" (по ключевым словам
   в тексте/атрибутах), вырезаем их текст.
4. Отправляем найденный текст в LLM (chat, без картинки) с вопросом:
   - это содержательный блок с бонусами и инструкцией, или
   - это просто ссылка/анонс, ведущий на другую страницу?
5. Если LLM говорит "это ссылка" — находим её href в этом же блоке,
   переходим по ней (goto) и повторяем шаги 2-4 на новой странице.
   Ограничение: не больше MAX_HOPS переходов, чтобы не зациклиться.
6. Если после переходов так и не нашли содержательный текстовый блок —
   fallback: делаем скриншот страницы и спрашиваем vision-моделью
   (chat_vision), что видно на картинке.
7. Валидатор (второй, независимый вызов LLM) проверяет финальный
   ответ на предмет галлюцинации.
8. Печатаем вердикт, программа завершается.

Использует pipeline_core.py (вся логика) и llm_api.py (методы работы
с API) и Playwright для браузера (готовый залогиненный профиль
./chrome_profile).

Установка (один раз):
    pip install playwright beautifulsoup4
    playwright install chromium

Запуск:
    python3 main.py --url "https://www.aeroflot.ru/ru-ru/bonus"

Если URL не задан - используется поиск в Google по QUERY (как в
step3_open_result.py) и открывается первый результат.
"""

import argparse
import configparser
import json
import os
import sys
import time
from datetime import datetime, timezone

import pipeline_core as pc
from pipeline_core import (
    RUN_STATS,
    SCRIPT_DIR,
    PROFILE_DIR,
    DEFAULT_QUERY_FILE,
    MAX_HOPS,
    MAX_CANDIDATE_SITES_TO_TRY,
    VIEWPORT_WIDTH,
    VIEWPORT_HEIGHT,
    VALIDATOR_SYSTEM_TMPL,
    log,
    silent_aware_print,
    load_query,
    extract_keywords_from_query,
    generate_search_query,
    goto_with_retry,
    looks_like_captcha_page,
    check_google_snippet_answer,
    process_candidate_site,
    make_give_up_summary,
    timed_chat,
)
import llm_api
from llm_api import LLMClient
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser(
        description="Универсальный поиск информации по заданной теме на "
                     "сайте: HTML -> LLM, с fallback на скриншот+vision. "
                     "Тема читается из текстового файла (по умолчанию "
                     "query.txt рядом со скриптом) - меняешь файл, скрипт "
                     "код трогать не нужно.")
    parser.add_argument("--url", default=None,
                        help="URL сайта. Если не задан - ищем в Google по теме из query.txt")
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE,
                        help=f"Путь к файлу с темой/запросом (по умолчанию {DEFAULT_QUERY_FILE})")
    parser.add_argument("--topic", default=None,
                        help="Тема/запрос ПРЯМО из командной строки, минуя "
                             "файл query.txt - например: "
                             "python3 main.py --topic \"погода в Казани 8 августа\". "
                             "Если задан - имеет приоритет над --query-file.")
    parser.add_argument("--config", default="config_vision.ini")
    parser.add_argument("--profile-dir", default=None,
                        help="Своя папка профиля Chrome (по умолчанию "
                             "chrome_profile рядом со скриптом). ОБЯЗАТЕЛЬНО "
                             "указывать разные значения, если запускаешь "
                             "несколько экземпляров main.py параллельно - "
                             "Chrome не даёт двум браузерам делить один и "
                             "тот же --user-data-dir одновременно (именно "
                             "так выглядит ошибка \"Opening in existing "
                             "browser session\").")
    args = parser.parse_args()

    # SCREENS_DIR теперь ВНУТРИ выбранного профиля Chrome, а не в общей
    # папке рядом со скриптом - иначе при параллельном запуске
    # нескольких main.py с разными --profile-dir все они делили бы одну
    # и ту же папку screens/, портя друг другу скриншоты.
    effective_profile_dir = args.profile_dir or PROFILE_DIR
    pc.SCREENS_DIR = os.path.join(effective_profile_dir, "screens")

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    # silent читается ИЗ КОНФИГА ДО первого log()/print() - иначе первые
    # строки (тема, ключевые слова) успели бы напечататься до того, как
    # мы узнали, что должны молчать.
    RUN_STATS["silent"] = cfg.getboolean("api", "silent", fallback=False)

    # Защита от бесконечного скролла (страницы с infinite scroll могут
    # генерировать контент бесконечно) - лимит настраивается через ини,
    # а не зашит в коде.
    pc.MAX_SCREENSHOTS = cfg.getint("browser", "max_scroll_screenshots", fallback=20)

    # Общий лимит времени на весь прогон (сек) - защита от ситуации
    # "проверили 3 сайта, каждый по 15-20 минут через vision, в сумме
    # больше часа" (реальный случай: прогон занял 3768 сек = 63 мин).
    # Если после текущей попытки уже превышен лимит - не начинаем
    # СЛЕДУЮЩУЮ попытку с новым сайтом, работаем с тем, что уже есть.
    max_total_runtime_sec = cfg.getint("browser", "max_total_runtime_sec", fallback=1800)

    if args.topic:
        topic = args.topic.strip()
        log(f"Тема взята напрямую из --topic (файл query.txt не читается).")
    else:
        topic = load_query(args.query_file)
    keywords = extract_keywords_from_query(topic)

    run_started_at = datetime.now(timezone.utc)
    t_run_start = time.monotonic()

    log(f"Тема поиска: \"{topic}\"")
    log(f"Ключевые слова для поиска блоков в HTML: {keywords}")

    # Чистим скриншоты от ПРЕДЫДУЩЕГО прогона сразу при старте, а не
    # только когда реально понадобится fallback на vision - иначе если
    # в этот раз fallback не понадобится, старые PNG остаются в папке
    # и путают, откуда взялся тот или иной файл.
    os.makedirs(pc.SCREENS_DIR, exist_ok=True)
    for f in os.listdir(pc.SCREENS_DIR):
        if f.startswith("screen_") and f.endswith(".png"):
            os.remove(os.path.join(pc.SCREENS_DIR, f))

    vision_client = LLMClient.from_config(cfg)  # секция [api] - для vision (chat_vision)

    def _proxy_status_str(c: LLMClient) -> str:
        """Явный индикатор для лога: пойдут ли запросы ЭТОГО клиента
        через SOCKS5-прокси. Без этого по одному тексту ошибки не
        всегда понятно, использовался ли прокси-путь (requests) или
        обычный прямой (urllib) - у них разный формат сообщений об
        ошибке, и это легко перепутать при диагностике."""
        if c.proxy_enabled and not c._is_local_host:
            return f"прокси: ДА (socks5h://{c.proxy_host}:{c.proxy_port})"
        elif c.proxy_enabled and c._is_local_host:
            return "прокси: НЕТ (хост локальный, прокси не применяется)"
        else:
            return "прокси: НЕТ (отключён в [proxy])"

    # Отдельный клиент для ТЕКСТОВЫХ запросов (поисковый запрос,
    # классификация ссылок, анализ HTML-фрагментов, финальная валидация).
    # Если секция [text_api] отсутствует в конфиге - используем ТОТ ЖЕ
    # клиент, что и для vision (обратная совместимость, ничего не
    # ломается для существующих конфигов без этой секции). Если секция
    # есть - это полностью независимый профиль: свой провайдер (например
    # облачный OpenRouter для текста, пока vision работает локально
    # через Ollama), свой active/local/remote, свои таймауты.
    if cfg.has_section("text_api"):
        text_client = LLMClient.from_config(cfg, config_section="text_api")
        silent_aware_print(
            f"Провайдер (vision): {vision_client.base_url}  модель: {vision_client.model}  "
            f"формат: {vision_client.api_format}  таймаут: {vision_client.timeout} сек  "
            f"{_proxy_status_str(vision_client)}")
        silent_aware_print(
            f"Провайдер (текст):  {text_client.base_url}  модель: {text_client.model}  "
            f"формат: {text_client.api_format}  таймаут: {text_client.timeout} сек  "
            f"{_proxy_status_str(text_client)}")
    else:
        text_client = vision_client
        silent_aware_print(f"Провайдер: {vision_client.base_url}  модель: {vision_client.model}  "
              f"формат: {vision_client.api_format}  таймаут запроса: {vision_client.timeout} сек  "
              f"{_proxy_status_str(vision_client)}")

    search_query_used = None  # заполняется ниже, если поиск через Google понадобился

    # SOCKS5-прокси для самого Chrome (не для API-запросов - те
    # настраиваются отдельно в LLMClient, см. api_enabled в
    # llm_api.py). browser_enabled управляет ТОЛЬКО браузером,
    # независимо от api_enabled - можно, например, гонять Chrome без
    # прокси (напрямую), а текстовые запросы к OpenRouter - через
    # прокси, или наоборот. Если browser_enabled не задан явно -
    # используется общий enabled (обратная совместимость).
    proxy_general_enabled = cfg.has_section("proxy") and cfg.getboolean("proxy", "enabled", fallback=False)
    proxy_enabled = (cfg.has_section("proxy")
                     and cfg.getboolean("proxy", "browser_enabled",
                                        fallback=proxy_general_enabled))
    proxy_host = cfg.get("proxy", "host", fallback="127.0.0.1")
    proxy_port = cfg.getint("proxy", "port", fallback=1080)

    # БАГ (был): таймаут запуска самого Chrome нигде не был настраиваем -
    # если запуск браузера подвисал (например конфликт с внешним
    # proxychains-ng, который оборачивает весь процесс python снаружи),
    # launch_persistent_context мог висеть БЕСКОНЕЧНО, единственный
    # выход - Ctrl+C. Теперь таймаут читается из [browser] секции ини,
    # в миллисекундах (как принято в Playwright).
    browser_launch_timeout_ms = cfg.getint("browser", "launch_timeout_ms", fallback=30000)
    navigation_timeout_ms = cfg.getint("browser", "navigation_timeout_ms", fallback=20000)

    browser_kwargs = dict(
        headless=False, channel="chrome",
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        timeout=browser_launch_timeout_ms,
        # БЕЗ --start-maximized: этот флаг заставляет окно занять
        # весь экран монитора, переопределяя заданный viewport -
        # тогда скриншоты снова были бы в разрешении монитора, а не
        # компактные 1024x768, как задумано выше.
        #
        # ignore_default_args убирает флаг "--enable-automation", который
        # Playwright добавляет САМ по умолчанию - именно из-за него Chrome
        # показывает предупреждение "браузер может быть небезопасен" при
        # входе в аккаунт Google, а Google чаще требует капчу (сайт видит
        # navigator.webdriver=true и другие явные признаки автоматизации).
        ignore_default_args=["--enable-automation"],
        args=[
            # Дополнительно скрывает флаг AutomationControlled в
            # navigator.webdriver, который многие антибот-системы
            # (включая Google) проверяют напрямую через JS.
            "--disable-blink-features=AutomationControlled",
            # Подавляет баннер "You are using an unsupported command-line
            # flag" - Chrome показывает его на ЛЮБОЙ нестандартный флаг,
            # включая --host-resolver-rules, который Playwright сам
            # генерирует из настройки proxy.bypass ("localhost,127.0.0.1")
            # - это не поломка и не наш флаг напрямую, просто честное
            # предупреждение Chrome, никак не влияющее на работу браузера.
            "--test-type",
        ],
    )
    if proxy_enabled:
        browser_kwargs["proxy"] = {
            "server": f"socks5://{proxy_host}:{proxy_port}",
            "bypass": "localhost,127.0.0.1",
        }
        log(f"Chrome запускается через SOCKS5-прокси {proxy_host}:{proxy_port}")

    log(f"Запускаю Chrome (таймаут запуска: {browser_launch_timeout_ms} мс)...")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            args.profile_dir or PROFILE_DIR, **browser_kwargs)
        # Таймаут по умолчанию для ВСЕХ последующих навигационных
        # операций (goto, wait_for_selector и т.п.), если явный timeout
        # не передан в конкретном вызове - тоже теперь из конфига,
        # а не зашит по всему коду разными хардкод-числами (было 10000,
        # 20000 в разных местах без единой настройки).
        context.set_default_navigation_timeout(navigation_timeout_ms)
        context.set_default_timeout(navigation_timeout_ms)
        page = context.pages[0] if context.pages else context.new_page()

        try:
            if args.url:
                # Явно указанный URL - без перебора кандидатов, пробуем
                # только его.
                final_answer, _ = process_candidate_site(
                    page, text_client, vision_client, topic, keywords,
                    args.url, already_loaded=False)
            else:
                # Не отправляем сырую тему из query.txt в Google как есть -
                # просим LLM саму сформулировать эффективный поисковый запрос
                # (без разговорных слов вроде "найди", "подскажи" и т.п.),
                # ничего не хардкодим.
                log("Прошу LLM сформулировать поисковый запрос по теме...")
                search_query = generate_search_query(text_client, topic)
                search_query_used = search_query
                log(f"Поисковый запрос от LLM: \"{search_query}\"")

                # Реальный случай: тема на русском ("разрешение экрана"), а
                # найденный сайт (например en.wikipedia.org) - на английском
                # ("screen resolution"). Русские ключевые слова НЕ совпадут с
                # английским текстом HTML, и find_bonus_blocks сразу проваливался
                # в vision-fallback, хотя данные могли быть прямо в тексте
                # страницы. Добавляем ключевые слова ИЗ АНГЛИЙСКОГО поискового
                # запроса тоже - теперь ищем совпадения на обоих языках сразу.
                extra_keywords = extract_keywords_from_query(search_query)
                keywords.extend(k for k in extra_keywords if k not in keywords)
                log(f"Ключевые слова дополнены словами из поискового запроса: {keywords}")

                goto_with_retry(page, "https://www.google.com")
                search_box = page.locator("textarea[name='q'], input[name='q']").first
                search_box.wait_for(state="visible", timeout=10000)
                search_box.fill(search_query)
                search_box.press("Enter")
                page.wait_for_selector("#search", timeout=10000)

                final_answer = None
                overall_sufficient = False
                sites_tried = 0

                # Сначала смотрим саму выдачу - вдруг ответ уже есть в
                # сниппете/AI Overview и открывать сайты вообще не придётся.
                snippet_answer = check_google_snippet_answer(
                    page, vision_client, text_client, topic, cfg)
                if snippet_answer is not None:
                    final_answer = snippet_answer
                    overall_sufficient = True
                    candidate_urls = []

                # Берём НЕСКОЛЬКО первых результатов - не только для
                # обхода капчи, но и для того, чтобы попробовать СЛЕДУЮЩИЙ
                # сайт целиком (текст+vision), если предыдущий не дал
                # достаточного ответа. Реальный случай: accuweather.com
                # не показал температуру на конкретную дату НИ В КАЛЕНДАРЕ,
                # НИ В ГРАФИКЕ (5 скриншотов, ~19 минут), хотя второй
                # кандидат (world-weather.ru с датой прямо в URL) мог дать
                # ответ быстрее и надёжнее - но пайплайн его даже не
                # пробовал, раньше сразу уходя на валидацию первого
                # неудачного результата.
                if snippet_answer is None:
                    result_links = page.locator("#search a:has(h3)")
                    result_count = min(result_links.count(), 5)
                    candidate_urls = []
                    for idx in range(result_count):
                        href = result_links.nth(idx).get_attribute("href")
                        if href:
                            candidate_urls.append(href)
                    log(f"Кандидаты из выдачи Google: {candidate_urls}")

                for idx, candidate_url in enumerate(candidate_urls, start=1):
                    if sites_tried >= MAX_CANDIDATE_SITES_TO_TRY:
                        log(f"Достигнут лимит в {MAX_CANDIDATE_SITES_TO_TRY} "
                            f"полностью проверенных сайта. Останавливаюсь "
                            f"на лучшем из найденных результатов.")
                        break

                    elapsed = time.monotonic() - t_run_start
                    if sites_tried > 0 and elapsed > max_total_runtime_sec:
                        # Уже потратили лимит времени (например каждый
                        # предыдущий сайт по 15-20 минут через vision) -
                        # не начинаем ЕЩЁ один долгий сайт, работаем с
                        # тем, что уже успели найти.
                        log(f"Превышен общий лимит времени на прогон "
                            f"({elapsed:.0f} сек > {max_total_runtime_sec} сек). "
                            f"Останавливаюсь на лучшем из найденных результатов.")
                        break

                    log(f"Пробую результат [{idx}]: {candidate_url}")
                    goto_with_retry(page, candidate_url)
                    page.wait_for_load_state("domcontentloaded", timeout=20000)
                    time.sleep(1)
                    if looks_like_captcha_page(page.content()):
                        log(f"Результат [{idx}] похож на страницу антибот-"
                            f"проверки (Cloudflare/капча) - решать её "
                            f"автоматически мы не пытаемся, пробую "
                            f"следующий результат из выдачи.")
                        continue

                    sites_tried += 1
                    site_answer, is_sufficient = process_candidate_site(
                        page, text_client, vision_client, topic, keywords,
                        candidate_url, already_loaded=True)

                    # Запоминаем ЛУЧШИЙ ответ на случай, если ни один
                    # сайт не окажется достаточным - лучше отдать хоть
                    # что-то, чем вообще ничего, но с явной пометкой
                    # недостаточности для валидатора/пользователя.
                    if final_answer is None or is_sufficient:
                        final_answer = site_answer

                    if is_sufficient:
                        log(f"Результат [{idx}] ({candidate_url}) дал "
                            f"достаточный ответ. Останавливаюсь.")
                        overall_sufficient = True
                        break
                    else:
                        log(f"Результат [{idx}] ({candidate_url}) не дал "
                            f"достаточного ответа. Пробую следующий сайт "
                            f"из выдачи, если есть...")

                if final_answer is None:
                    log("Все проверенные результаты поиска оказались "
                        "защищены капчей или недоступны. Останавливаюсь.")
                    # context.close() НЕ вызываем здесь явно - finally
                    # ниже гарантированно вызовет его сам после return
                    # (закрытие уже закрытого контекста дважды могло бы
                    # выдать ошибку).
                    write_result_and_exit(
                        cfg, topic=topic, search_query=search_query_used,
                        started_at=run_started_at, t_run_start=t_run_start,
                        success=False, final_answer=None,
                        validator_verdict=None, error="all search results were captcha/unavailable")
                    return

                if not overall_sufficient:
                    log("Ни один из проверенных сайтов не дал полностью "
                        "достаточного ответа - сжимаю в короткое честное "
                        "резюме вместо того, чтобы отдавать сырую простыню "
                        "из нескольких длинных описаний.")
                    raw_notes = final_answer  # то, что успели собрать
                    final_answer = make_give_up_summary(
                        text_client, topic, raw_notes, sites_tried)
        finally:
            # Chrome закрывается ВСЕГДА, даже если внутри run_pipeline
            # или vision-цикла вылетело исключение.
            context.close()

    # --- Итоговый ответ пользователю (баг: раньше выводился только
    # вердикт валидатора, а сам найденный ответ по теме нигде явно не
    # печатался - терялся в потоке логов промежуточных скриншотов) ---
    silent_aware_print("\n" + "=" * 60)
    silent_aware_print(f"ИТОГОВЫЙ ОТВЕТ по теме \"{topic}\":")
    silent_aware_print("=" * 60)
    silent_aware_print(final_answer)
    silent_aware_print("=" * 60)

    # --- Валидатор ---
    log("-" * 60)
    log("Проверка финального ответа валидатором...")
    validator_answer = timed_chat(
        text_client,
        system=VALIDATOR_SYSTEM_TMPL.format(topic=topic),
        user=f"Итоговый ответ для проверки:\n\n---\n{final_answer}\n---\n\n"
             "Дай вывод в формате:\n"
             "ВЕРДИКТ: похоже на достоверный анализ / похоже на галлюцинацию\n"
             "ПРИЧИНА: <короткое обоснование>",
    )
    silent_aware_print("\nВердикт валидатора:")
    silent_aware_print(validator_answer)
    silent_aware_print("-" * 60)
    silent_aware_print("Готово. Программа завершена.")

    # БАГ (был): success=True ставился ВСЕГДА при отсутствии исключений,
    # даже если сам валидатор явно написал "похоже на галлюцинацию" -
    # success означал только "скрипт не упал", а не "ответ достоверен".
    #
    # БАГ (был №2): наивная проверка "галлюцинац" in весь_текст.lower()
    # ловила ложные срабатывания - реальный случай: валидатор написал
    # "Признаков галлюцинаций ... нет" (то есть ОТРИЦАНИЕ галлюцинации,
    # сам вердикт был "похоже на достоверный анализ"), но подстрока
    # "галлюцинац" всё равно встречается в тексте ПРИЧИНЫ - success
    # ошибочно ставился в False. Теперь проверяем ТОЛЬКО строку с
    # "ВЕРДИКТ:", а не весь текст вместе с обоснованием.
    verdict_line = next(
        (line for line in validator_answer.splitlines()
         if line.strip().upper().startswith("ВЕРДИКТ")),
        validator_answer,  # если формат вдруг не соблюдён - fallback на весь текст
    )
    validator_says_hallucination = "галлюцинац" in verdict_line.lower()
    success = not validator_says_hallucination

    write_result_and_exit(
        cfg, topic=topic, search_query=search_query_used,
        started_at=run_started_at, t_run_start=t_run_start,
        success=success, final_answer=final_answer,
        validator_verdict=validator_answer,
        error=None if success else "validator flagged result as hallucination")


def write_result_and_exit(cfg, topic, search_query, started_at, t_run_start,
                          success, final_answer, validator_verdict, error):
    """Собирает итоговый result.json (для последующей обработки другой
    программой/AI) и либо сохраняет его в файл, либо (в silent-режиме)
    только печатает JSON в консоль и ничего не сохраняет на диск -
    решается флагом silent = true/false в [api] секции config_vision.ini."""
    finished_at = datetime.now(timezone.utc)
    duration_sec = round(time.monotonic() - t_run_start, 1)

    result = {
        "topic": topic,
        "search_query_used": search_query,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_sec,
        "llm_requests_total": RUN_STATS["llm_requests_total"],
        "token_usage": llm_api.get_token_stats_or_none(),
        "success": success,
        "final_answer": final_answer,
        "validator_verdict": validator_verdict,
        "error": error,
    }
    result_json = json.dumps(result, ensure_ascii=False, indent=2)

    if RUN_STATS["silent"]:
        # silent = true -> ТОЛЬКО JSON в консоль, ничего больше,
        # никакого файла на диске.
        print(result_json)
    else:
        result_path = os.path.join(SCRIPT_DIR, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        silent_aware_print(f"\nРезультат сохранён в {result_path}")
        silent_aware_print(result_json)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
