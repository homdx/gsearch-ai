"""
ШАГ 1: Открыть настоящий Chrome с постоянным профилем для логина.
Логинишься один раз вручную (Google и т.д.) — сессия сохранится в папке
профиля и будет доступна всем следующим запускам main.py.

СОГЛАСОВАНО с текущими настройками main.py:
  - те же антидетект-флаги (иначе Google может показывать "браузер
    небезопасен" здесь при логине, а потом main.py откроет уже другой,
    "чистый" браузер без этой защиты - несогласованность бессмысленна)
  - тот же viewport (1024x768) - логиниться удобнее в реальном размере
    окна, а не в maximized, который потом всё равно не используется
  - поддержка --profile-dir - для логина в НЕСКОЛЬКО параллельных
    профилей (см. main.py --profile-dir)
  - опциональная поддержка прокси из config_vision.ini - если сайт для
    логина недоступен без прокси, включи так же, как в основном скрипте

Проверка перед первым запуском (Ubuntu), нужен установленный Chrome:
    google-chrome --version
Если не установлен:
    sudo apt install google-chrome-stable

Установка Playwright (один раз):
    pip install playwright
    playwright install chromium

Запуск:
    python3 step1_open_chrome.py
    python3 step1_open_chrome.py --profile-dir chrome_profile2
    python3 step1_open_chrome.py --url https://www.aeroflot.ru
"""

import argparse
import configparser
import os

from playwright.sync_api import sync_playwright

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Те же значения, что и в main.py - если поменяешь там, поменяй и тут,
# чтобы логин происходил в том же окружении, в котором потом реально
# работает основной скрипт.
DEFAULT_PROFILE_DIR = os.path.join(SCRIPT_DIR, "chrome_profile")
VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 768


def main():
    parser = argparse.ArgumentParser(
        description="Открыть Chrome с постоянным профилем для ручного логина")
    parser.add_argument("--profile-dir", default=None,
                        help=f"Папка профиля (по умолчанию {DEFAULT_PROFILE_DIR}). "
                             "Используй ТУ ЖЕ папку, что будешь передавать в "
                             "main.py --profile-dir, если запускаешь несколько "
                             "параллельных профилей.")
    parser.add_argument("--url", default="https://www.google.com",
                        help="Какой сайт открыть сразу после запуска (по умолчанию Google)")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config_vision.ini"),
                        help="Путь к конфигу - используется только для чтения "
                             "настроек прокси, если он там включён")
    args = parser.parse_args()

    profile_dir = args.profile_dir or DEFAULT_PROFILE_DIR

    # Прокси читаем из ТОГО ЖЕ конфига, что и main.py, чтобы логиниться
    # в том же сетевом окружении, в котором потом реально будет работать
    # основной скрипт (например если нужный сайт недоступен без прокси).
    proxy_enabled = False
    proxy_host, proxy_port = "127.0.0.1", 1080
    if os.path.exists(args.config):
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        if cfg.has_section("proxy"):
            general_enabled = cfg.getboolean("proxy", "enabled", fallback=False)
            proxy_enabled = cfg.getboolean("proxy", "browser_enabled", fallback=general_enabled)
            proxy_host = cfg.get("proxy", "host", fallback=proxy_host)
            proxy_port = cfg.getint("proxy", "port", fallback=proxy_port)

    browser_kwargs = dict(
        headless=False,
        channel="chrome",              # использовать настоящий Chrome, не Chromium
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        # БЕЗ --start-maximized: раньше был здесь, но он подменяет
        # реальный размер окна на разрешение монитора - если потом
        # main.py снимает скриншоты в 1024x768, логиниться удобнее
        # сразу в этом же размере, а не в maximized.
        #
        # ignore_default_args убирает флаг "--enable-automation",
        # который Playwright добавляет сам по умолчанию - именно из-за
        # него Chrome показывает баннер "браузер может быть небезопасен"
        # при попытке войти в аккаунт Google, что мешает логину именно
        # на этом самом шаге.
        ignore_default_args=["--enable-automation"],
        args=[
            # Скрывает navigator.webdriver=true от JS-проверок сайтов
            # (антибот-системы, включая Google, часто проверяют это
            # напрямую) - без этого чаще требуется капча даже при
            # обычном ручном логине.
            "--disable-blink-features=AutomationControlled",
            # Подавляет баннер "unsupported command-line flag" - Chrome
            # показывает его на любой нестандартный флаг (в т.ч. на тот,
            # что появляется при использовании прокси с bypass-списком),
            # это не ошибка и не мешает работе, просто визуальный шум.
            "--test-type",
        ],
    )

    if proxy_enabled:
        browser_kwargs["proxy"] = {
            "server": f"socks5://{proxy_host}:{proxy_port}",
            "bypass": "localhost,127.0.0.1",
        }
        print(f"Chrome запускается через SOCKS5-прокси {proxy_host}:{proxy_port}")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(profile_dir, **browser_kwargs)

        # Берём первую вкладку, если она уже есть, иначе создаём новую
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(args.url)

        print(f"Chrome открыт с профилем: {profile_dir}")
        print("Залогинься вручную во всё, что нужно (Google и т.д.)")
        input("Когда закончишь — нажми Enter здесь, чтобы закрыть браузер...")

        page.close()
        context.close()
        print(f"Сессия сохранена в {profile_dir}")


if __name__ == "__main__":
    main()
