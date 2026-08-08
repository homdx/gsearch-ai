"""
llm_api.py — универсальный клиент для LLM API.

Только методы работы с API. Никакой игровой логики (взято за основу
из llm_client.py репозитория learn-in-play1, но всё игровое: _Breaker,
_ServerCaps, промиз-леджеры и т.п. — выпилено).

Поддерживает:
  - api_format = "ollama"  -> локальный `ollama serve` (офлайн)
  - api_format = "openai"  -> любой OpenAI-совместимый сервер (онлайн):
    OpenAI, OpenRouter, Mistral, Groq, HuggingFace router и т.д.

Методы:
  LLMClient.from_config(cfg)   -> создать клиента из config.ini
  client.chat(system, user)    -> обычный текстовый запрос
  client.chat_vision(prompt, image_path) -> запрос с картинкой
"""

import base64
import json
import os
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from urllib.parse import urlparse

# requests - используется ТОЛЬКО для запросов через прокси (обычный
# urllib-путь без прокси не тронут, работает как раньше). Через прокси
# используем requests, а не монки-патч socket.socket - монки-патч
# ненадёжен для HTTPS-запросов через urllib (DNS может резолвиться
# напрямую в обход прокси даже при подменённом socket.socket, из-за
# чего внешний облачный API видит настоящий IP и отдаёт 403, хотя
# кажется, что прокси включён). requests + PySocks с схемой "socks5h"
# резолвит DNS ЧЕРЕЗ сам прокси - без этой утечки.
try:
    import requests
except ImportError:
    requests = None

# PySocks - опциональная зависимость, нужна ТОЛЬКО если реально включаешь
# SOCKS5-прокси (proxy_enabled=True при создании клиента). Без неё весь
# остальной модуль работает как раньше (прямые запросы без прокси).
try:
    import socks  # noqa: F401 - импортируется requests неявно, здесь только для проверки наличия
    _HAS_PYSOCKS = True
except ImportError:
    _HAS_PYSOCKS = False

# Хосты, для которых прокси НИКОГДА не используется, даже если прокси
# включён глобально - локальный Ollama (или любой другой локальный
# сервис) должен быть доступен напрямую, прокси тут не нужен и часто
# даже не может до него достучаться (SOCKS-прокси обычно настроен на
# выход в интернет, а не на петлю до localhost).
NO_PROXY_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Накопитель реально потреблённых токенов за весь прогон - ТОЛЬКО из
# того, что провайдер сам прислал в ответе (поле "usage" у OpenAI-
# совместимых API, "prompt_eval_count"/"eval_count" у Ollama). Ничего
# не запрашивается у модели дополнительно и не требуется в промптах -
# если провайдер это поле не прислал (бывает у некоторых бесплатных/
# урезанных эндпоинтов), просто не считаем, никаких попыток угадать.
TOKEN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "requests_with_usage": 0,   # на скольких ответах usage реально был
}


def get_token_stats_or_none() -> dict | None:
    """Снимок TOKEN_STATS для итогового JSON. Возвращает None, если ни
    один ответ за весь прогон не содержал usage/eval-полей - чтобы в
    JSON не появлялось лживое "tokens_total": 0, будто токены посчитаны,
    когда на самом деле провайдер их просто не присылал."""
    if TOKEN_STATS["requests_with_usage"] == 0:
        return None
    return dict(TOKEN_STATS)


def _accumulate_usage(raw: dict, api_format: str) -> None:
    """Пополняет TOKEN_STATS данными из одного raw-ответа API, если они
    там есть. Молча ничего не делает, если usage/eval-полей нет."""
    if api_format == "ollama":
        prompt = raw.get("prompt_eval_count")
        completion = raw.get("eval_count")
        if prompt is None and completion is None:
            return
        prompt = prompt or 0
        completion = completion or 0
        TOKEN_STATS["prompt_tokens"] += prompt
        TOKEN_STATS["completion_tokens"] += completion
        TOKEN_STATS["total_tokens"] += prompt + completion
        TOKEN_STATS["requests_with_usage"] += 1
    else:
        usage = raw.get("usage")
        if not usage:
            return
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        total = usage.get("total_tokens")
        if total is None:
            total = prompt + completion
        TOKEN_STATS["prompt_tokens"] += prompt
        TOKEN_STATS["completion_tokens"] += completion
        TOKEN_STATS["total_tokens"] += total
        TOKEN_STATS["requests_with_usage"] += 1


def strip_think(text: str) -> str:
    """Вырезает <think>...</think>, если модель вернула reasoning-блок."""
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def _make_unverified_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ollama_chat_url(base_url: str) -> str:
    return f"{base_url}/api/chat"


def _extract_error_message(raw: dict) -> str:
    """OpenRouter (и многие OpenAI-совместимые шлюзы) на некоторые сбои
    (дневной лимит бесплатной модели, недоступность конкретного
    провайдера, слишком длинный контекст и т.п.) отвечают HTTP 200 OK,
    но с телом {"error": {...}} вместо нормального ответа с "choices".
    Без этой проверки код лез прямо в raw["choices"][0]... и падал с
    невнятным KeyError - вместо этого достаём текст ошибки, если он
    там есть."""
    err = raw.get("error")
    if err is None:
        return None
    if isinstance(err, dict):
        return err.get("message") or str(err)
    return str(err)


def _debug_from_env() -> bool:
    """Переменная окружения LLM_DEBUG=1 (или true/yes) - быстрый способ
    включить подробный лог запросов БЕЗ правки config_vision.ini,
    сразу для ВСЕХ клиентов (vision и text). Реальный случай: пользователь
    запускал `LLM_DEBUG=1 python3 main.py ...`, ожидая debug-вывод, но
    переменная нигде не читалась - код проверял только debug=true в
    конфиге. Теперь env-переменная - это OVERRIDE, включает debug, даже
    если в конфиге стоит false (но не может ВЫКЛЮЧИТЬ debug, если он
    явно включён в конфиге - это только "добавочный" способ включения)."""
    return os.environ.get("LLM_DEBUG", "").strip().lower() in ("1", "true", "yes")


class _SoftAPIError(Exception):
    """"Мягкая" ошибка: HTTP-статус 200 (запрос технически прошёл), но
    тело ответа - {"error": ...} вместо нормального результата.
    Реальный случай: OpenRouter/Nvidia отдали "ResourceExhausted:
    Worker local total request limit reached (36/32)" - временная
    перегрузка воркера у провайдера модели, которая с высокой
    вероятностью пройдёт при повторе через несколько секунд, а не
    смертельный сбой. Раньше такие ошибки сразу поднимались наружу
    без единой попытки повтора (в отличие от HTTP-уровневых ошибок
    429/5xx, для которых retry уже был) - выделяем в отдельный класс
    исключений, чтобы _post() мог обрабатывать их ТЕМ ЖЕ retry-циклом,
    что и обычные сетевые/HTTP сбои."""
    pass


class LLMClient:
    """Обёртка над одним профилем API (секция [api_local] или [api_remote]
    в config.ini). Поддерживает api_format "ollama" и "openai"."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 api_format: str = "ollama", verify_ssl: bool = True,
                 num_ctx: int = 0, think: "bool | None" = None,
                 timeout: int = 120, timeout_vision: int = None,
                 retries: int = 1,
                 error_retries: int = 2, error_retry_wait_sec: int = 30,
                 proxy_enabled: bool = False, proxy_host: str = "127.0.0.1",
                 proxy_port: int = 1080,
                 context_budget_tokens: int = 0,
                 debug: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        # debug=True включает подробный лог запросов/ответов в стиле
        # оригинального hallucination_test.py: таймстампы отправки и
        # получения, полное сырое тело ответа сервера (json.dumps с
        # indent=2) - полезно для диагностики, что именно возвращает
        # конкретный провайдер, но слишком многословно для обычной
        # работы (поэтому выключено по умолчанию).
        self.debug = debug
        # Vision-запросы (картинка) обычно заметно дольше текстовых,
        # особенно на CPU. Если отдельный таймаут не задан в конфиге -
        # используем тот же, что и для текста.
        self.timeout_vision = timeout_vision if timeout_vision else timeout
        self.retries = max(0, int(retries))
        self.error_retries = max(0, int(error_retries))
        self.error_retry_wait_sec = max(1, int(error_retry_wait_sec))
        self._ssl_context = None if verify_ssl else _make_unverified_context()
        # SOCKS5-прокси - применяется в _post() ТОЛЬКО если base_url не
        # localhost (см. NO_PROXY_HOSTS) и proxy_enabled=True. Так
        # локальный Ollama (обычно localhost:11434) всегда доступен
        # напрямую, даже когда прокси включён для удалённых провайдеров.
        self.proxy_enabled = proxy_enabled
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self._is_local_host = urlparse(self.base_url).hostname in NO_PROXY_HOSTS
        # context_budget_tokens - явный бюджет контекста для УДАЛЁННЫХ
        # (не-ollama) провайдеров, у которых нет num_ctx (это чисто
        # ollama-параметр). Без него любой remote-клиент (OpenRouter и
        # т.п.) не имел понятия о своём реальном контекстном окне -
        # используется вызывающим кодом, чтобы решить, нужно ли резать
        # длинный текст на несколько частей и делать НЕСКОЛЬКО заходов
        # в LLM вместо одного обрезанного запроса.
        self.context_budget_tokens = context_budget_tokens

    @property
    def input_char_budget(self) -> int:
        """Сколько символов ВХОДНОГО текста можно безопасно передать в
        одном запросе, с учётом контекстного окна модели (num_ctx для
        ollama, context_budget_tokens для remote) и запаса на системный
        промпт + место для ответа. Грубая оценка ~4 символа на токен
        (усреднённо для русского/английского текста) - точный подсчёт
        токенов требует токенизатора конкретной модели, которого у нас
        нет, но с запасом ~25% этого достаточно, чтобы не обрезать
        текст в неудачном месте и не превысить лимит модели."""
        total_tokens = self.num_ctx or self.context_budget_tokens or 8000
        # Резервируем часть под системный промпт, инструкцию и ответ -
        # без этого можно превысить лимит контекста самим "обрамлением"
        # запроса, даже если основной текст впритык укладывается.
        reserved_tokens = min(1500, total_tokens // 4)
        usable_tokens = max(500, total_tokens - reserved_tokens)
        return usable_tokens * 4  # ~4 символа на токен, грубая оценка


    @classmethod
    def from_config(cls, cfg, active: str = None, config_section: str = "api") -> "LLMClient":
        """Строит клиента из configparser.ConfigParser.
        [<config_section>].active указывает, какую секцию
        <config_section>_<active> использовать.

        config_section по умолчанию "api" (обратная совместимость -
        существующие конфиги без изменений). Передав, например,
        config_section="text_api", можно завести ПОЛНОСТЬЮ независимый
        профиль для текстовых запросов (свой active/local/remote/
        таймауты), отдельно от vision - например, vision локально через
        Ollama, а текстовый анализ/валидация - в облако через OpenRouter.

        Прокси читается из ОБЩЕЙ секции [proxy] (одна на весь конфиг,
        не привязана к config_section). api_enabled управляет ТОЛЬКО
        API-запросами (LLMClient) независимо от enabled_browser для
        Chrome - можно включить прокси для одного и выключить для
        другого. Если api_enabled не задан явно - используется общий
        enabled (обратная совместимость со старыми конфигами)."""
        active = active or cfg.get(config_section, "active", fallback="local")
        section = f"{config_section}_{active}"
        verify_ssl = cfg.getboolean(config_section, "verify_ssl", fallback=True)

        proxy_general_enabled = (cfg.has_section("proxy")
                                 and cfg.getboolean("proxy", "enabled", fallback=False))
        proxy_enabled = (cfg.has_section("proxy")
                         and cfg.getboolean("proxy", "api_enabled",
                                            fallback=proxy_general_enabled))
        proxy_host = cfg.get("proxy", "host", fallback="127.0.0.1")
        proxy_port = cfg.getint("proxy", "port", fallback=1080)

        return cls(
            base_url=cfg.get(section, "base_url"),
            api_key=cfg.get(section, "api_key", fallback="not-needed"),
            model=cfg.get(section, "model"),
            api_format=cfg.get(section, "api_format", fallback="ollama"),
            num_ctx=cfg.getint(section, "num_ctx", fallback=0),
            think=(None if cfg.get(section, "think", fallback=None) is None
                   else cfg.getboolean(section, "think")),
            verify_ssl=verify_ssl,
            timeout=cfg.getint(config_section, "timeout_seconds", fallback=120),
            timeout_vision=cfg.getint(config_section, "timeout_seconds_vision",
                                      fallback=0) or None,
            retries=cfg.getint(config_section, "retries", fallback=1),
            error_retries=cfg.getint(config_section, "error_retries", fallback=2),
            error_retry_wait_sec=cfg.getint(config_section, "error_retry_wait_sec",
                                             fallback=30),
            proxy_enabled=proxy_enabled,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            context_budget_tokens=cfg.getint(section, "context_budget_tokens", fallback=0),
            debug=cfg.getboolean(config_section, "debug", fallback=False) or _debug_from_env(),
        )

    # ------------------------------------------------------------------
    # Внутреннее: единый POST-запрос с повторами на сетевых/HTTP ошибках
    # ------------------------------------------------------------------
    def _post(self, url: str, payload: dict, timeout: int = None, attempt: int = 0) -> dict:
        timeout = timeout if timeout is not None else self.timeout
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # UA-1 (тот же случай, что был в llm_client.py репозитория
            # learn-in-play1): urllib/requests без "человеческого"
            # User-Agent часто банится Cloudflare-защитой (у OpenRouter
            # именно она) ещё до JS-челленджа, просто по сигнатуре
            # известных библиотечных строк - "Access denied by security
            # policy" это типичный текст такого блока. Строка с URL в
            # скобках обычно достаточна, чтобы пройти именно этот фильтр.
            "User-Agent": "llm-api-client/1.0 (+https://github.com/homdx/learn-in-play1)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        }

        # Прокси используется ТОЛЬКО для удалённых хостов - локальный
        # Ollama (localhost/127.0.0.1) всегда идёт напрямую, даже если
        # прокси включён глобально в [proxy] секции конфига.
        use_proxy = self.proxy_enabled and not self._is_local_host

        if self.debug:
            t_start = datetime.now(timezone.utc)
            print(f"\n🕒 Запрос отправлен: {t_start.isoformat(timespec='milliseconds')}")

        try:
            if use_proxy:
                raw = self._post_via_requests(url, payload, headers, timeout)
            else:
                raw = self._post_via_urllib(url, payload, headers, timeout)

            if self.debug:
                t_end = datetime.now(timezone.utc)
                print(f"🕒 Ответ получен: {t_end.isoformat(timespec='milliseconds')} "
                      f"(прошло {(t_end - t_start).total_seconds():.1f}s)")
                print("\n[ПОЛНЫЙ RAW-ОТВЕТ СЕРВЕРА]:")
                print(json.dumps(raw, ensure_ascii=False, indent=2))

            # "Мягкая" ошибка - HTTP 200, но тело {"error": ...} вместо
            # результата (например временная перегрузка воркера у
            # провайдера модели). Заворачиваем в _SoftAPIError, чтобы
            # ниже она попала в ТОТ ЖЕ retry-цикл, что и обычные
            # сетевые/HTTP сбои - раньше такие ошибки сразу падали без
            # единой попытки повтора.
            error_msg = _extract_error_message(raw)
            if error_msg:
                raise _SoftAPIError(error_msg)
            return raw
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, OSError, _SoftAPIError) as e:
            if attempt < self.error_retries:
                log_msg = (f"Попытка {attempt + 1}/{self.error_retries + 1} для "
                          f"{url} не удалась ({e}), повтор через "
                          f"{self.error_retry_wait_sec} сек...")
                print(log_msg)
                time.sleep(self.error_retry_wait_sec)
                return self._post(url, payload, timeout=timeout, attempt=attempt + 1)
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
            elif requests is not None and isinstance(e, requests.exceptions.HTTPError):
                try:
                    detail = e.response.text[:500]
                except Exception:
                    pass
            elif isinstance(e, _SoftAPIError):
                detail = str(e)
            raise RuntimeError(
                f"Ошибка запроса к {url} после {attempt + 1} попыток "
                f"(модель {self.model}): {e} {detail}") from None

    def _post_via_urllib(self, url: str, payload: dict, headers: dict, timeout: int) -> dict:
        """Прямой запрос без прокси - оригинальный путь через urllib,
        не тронут, работает как и раньше."""
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout,
                                     context=self._ssl_context) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_via_requests(self, url: str, payload: dict, headers: dict, timeout: int) -> dict:
        """Запрос ЧЕРЕЗ SOCKS5-прокси, используя requests + PySocks со
        схемой "socks5h" (а не "socks5") - буква "h" здесь принципиальна:
        она заставляет резолвить DNS-имя хоста ЧЕРЕЗ сам прокси, а не
        локально до подключения. Без "h" (или при монки-патче
        socket.socket, как было раньше) DNS может уйти напрямую в обход
        прокси, и облачный API увидит настоящий IP клиента и вернёт 403,
        хотя визуально кажется, что прокси используется."""
        if requests is None:
            raise RuntimeError(
                "Для работы через прокси нужна библиотека requests. Установи:\n"
                "  pip install requests --break-system-packages")
        if not _HAS_PYSOCKS:
            raise RuntimeError(
                "Для SOCKS5-прокси нужна библиотека PySocks. Установи:\n"
                "  pip install pysocks --break-system-packages")

        proxy_url = f"socks5h://{self.proxy_host}:{self.proxy_port}"
        proxies = {"http": proxy_url, "https": proxy_url}

        # self._ssl_context is None означает verify_ssl=True (обычная
        # проверка сертификата); если verify_ssl=False, там лежит
        # специально созданный "небезопасный" контекст - для requests
        # это транслируется в verify=False.
        verify_ssl_for_requests = self._ssl_context is None

        resp = requests.post(
            url, headers=headers, json=payload, timeout=timeout,
            proxies=proxies, verify=verify_ssl_for_requests,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Текстовый чат
    # ------------------------------------------------------------------
    def _is_truncated(self, raw: dict) -> bool:
        """Проверяет finish_reason/done_reason - модель остановилась
        потому что закончила мысль (stop) или потому что упёрлась в
        max_tokens (length)? Реальный случай: ответ обрывался на
        полуслове ("...задержи"), валидатор справедливо называл это
        галлюцинацией/битой генерацией - хотя проблема была не в
        содержании, а просто в том, что не хватило места дописать."""
        if self.api_format == "ollama":
            return raw.get("done_reason") == "length"
        else:
            choices = raw.get("choices") or [{}]
            return choices[0].get("finish_reason") == "length"

    def chat(self, system: str, user: str, temperature: float = 0.3,
             max_tokens: int = 800, max_continuations: int = 2,
             disable_reasoning: bool = False) -> str:
        """max_continuations - сколько ДОПОЛНИТЕЛЬНЫХ вызовов делать,
        если ответ обрывается по лимиту токенов (finish_reason/
        done_reason == "length"). Модели дают инструкцию продолжить
        ТОЧНО с места обрыва, ответы склеиваются - вместо того чтобы
        отдавать наружу заведомо битый, обрубленный на полуслове текст.

        disable_reasoning=True - для OpenRouter (и совместимых) шлётся
        "reasoning": {"exclude": true}, чтобы модель с thinking-режимом
        (например nemotron) не тратила весь лимит токенов на служебное
        вступление к рассуждению вместо самого ответа. Реальный случай:
        классификация URL с max_tokens=10 упиралась в finish_reason=
        "length" на фразе "The user wants me to classify a URL based
        on" - токены полностью уходили на начало рассуждения, а до
        VALID/INVALID дело не доходило вообще, и цикл докрутки
        (max_continuations) не спасал, т.к. каждый новый вызов заново
        начинал такое же длинное вступление. Используй True для
        коротких классифицирующих запросов (VALID/INVALID, да/нет)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        full_content = ""
        for continuation_round in range(max_continuations + 1):
            if self.api_format == "ollama":
                url = _ollama_chat_url(self.base_url)
                options = {"temperature": temperature, "num_predict": max_tokens}
                if self.num_ctx:
                    options["num_ctx"] = self.num_ctx
                payload = {"model": self.model, "messages": messages,
                           "stream": False, "options": options}
                if self.think is not None:
                    payload["think"] = self.think
                if disable_reasoning:
                    payload["think"] = False
            else:
                url = f"{self.base_url}/chat/completions"
                payload = {
                    "model": self.model, "temperature": temperature,
                    "max_tokens": max_tokens, "messages": messages,
                    "stream": False,
                }
                if disable_reasoning:
                    # OpenRouter-специфичный параметр - полностью убирает
                    # генерацию reasoning-токенов для моделей, которые
                    # это поддерживают. Для моделей без такой поддержки
                    # просто игнорируется провайдером, безопасно слать
                    # всегда, когда нужен короткий прямой ответ.
                    payload["reasoning"] = {"exclude": True}

            raw = self._post(url, payload, timeout=self.timeout)
            # Проверка на ошибку теперь внутри _post() (с retry) - сюда
            # raw попадает уже гарантированно без "error" в теле, либо
            # _post() выбросил бы исключение после исчерпания попыток.
            _accumulate_usage(raw, self.api_format)
            content_field = (raw.get("message", {}).get("content", "")
                             if self.api_format == "ollama"
                             else raw["choices"][0]["message"].get("content", ""))
            piece = strip_think(content_field)

            # БАГ (был): некоторые модели (например nemotron с thinking-
            # режимом) кладут ВЕСЬ содержательный анализ в поле
            # "reasoning"/"reasoning_details", а "content" остаётся
            # пустым или содержит только служебный обрывок. Раньше это
            # приводило к молчаливой потере реального ответа (валидные
            # цифры/факты были в reasoning, но код их даже не смотрел).
            #
            # РЕГРЕСС (был после первого фикса): условие "reasoning
            # длиннее content в 2+ раза" оказалось СЛИШКОМ агрессивным -
            # у reasoning-моделей поле reasoning почти ВСЕГДА длиннее
            # content, даже когда content уже является ПОЛНОЦЕННЫМ
            # правильным ответом (например "СОДЕРЖАТЕЛЬНЫЙ БЛОК\n<факты>"
            # - короткий и корректный), а reasoning - это просто
            # многословные внутренние рассуждения модели перед ответом
            # ("I need to provide a brief summary"), НЕ являющиеся
            # готовым ответом. Реальный случай: content был правильным
            # структурированным ответом, но эвристика подменила его
            # длинным текстом рассуждений, испортив итоговый результат.
            #
            # Теперь используем reasoning ТОЛЬКО если content
            # практически ПУСТ (не просто короче reasoning) - это и был
            # исходный мотивирующий случай (пустой/обрывочный content
            # при непустом reasoning), а не "content короче reasoning".
            if self.api_format != "ollama" and not disable_reasoning:
                reasoning_field = raw["choices"][0]["message"].get("reasoning") or ""
                if reasoning_field and len(piece.strip()) < 5:
                    piece = strip_think(reasoning_field)

            full_content += piece

            if not self._is_truncated(raw) or continuation_round >= max_continuations:
                break

            # Продолжаем диалог: добавляем то, что модель УЖЕ сгенерировала
            # (как assistant-сообщение), и просим дописать остаток. Это
            # штатный способ "допроса" в chat-формате - модель видит свой
            # же обрубленный текст и продолжает именно с этого места, а
            # не начинает заново.
            messages.append({"role": "assistant", "content": piece})
            messages.append({"role": "user", "content":
                            "Продолжи ТОЧНО с того места, где твой предыдущий "
                            "ответ был прерван - не повторяй уже написанное, "
                            "не начинай заново, просто допиши остаток."})

        return full_content

    # ------------------------------------------------------------------
    # Чат с картинкой (vision)
    # ------------------------------------------------------------------
    def chat_vision(self, prompt: str, image_path: str,
                    temperature: float = 0.2, max_tokens: int = 800) -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        if self.api_format == "ollama":
            url = _ollama_chat_url(self.base_url)
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt,
                             "images": [image_b64]}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            # БАГ (был): think здесь не передавался, из-за чего модель с
            # включённым по умолчанию thinking-режимом тратила ВЕСЬ
            # max_tokens на рассуждения в <think>...</think> и не успевала
            # написать сам ответ - на выходе пустая строка после strip_think.
            if self.think is not None:
                payload["think"] = self.think
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }],
            }
            if self.think is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.think}

        raw = self._post(url, payload, timeout=self.timeout_vision)
        # Проверка на ошибку теперь внутри _post() (с retry).
        _accumulate_usage(raw, self.api_format)
        content = (raw.get("message", {}).get("content", "")
                   if self.api_format == "ollama"
                   else raw["choices"][0]["message"]["content"])
        if not content.strip():
            # Явный сигнал, что модель вернула пустоту (обычно значит:
            # весь max_tokens ушёл на скрытые рассуждения) - не молчим.
            return "[МОДЕЛЬ ВЕРНУЛА ПУСТОЙ ОТВЕТ - возможно, не хватило max_tokens]"
        return strip_think(content)
