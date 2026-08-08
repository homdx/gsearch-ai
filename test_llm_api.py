"""
test_llm_api.py — регрессионные тесты для llm_api.py.

Это реальные проверки, которые делались вручную в консоли (python3 -c "...")
по ходу работы над скриптом, теперь оформленные как pytest-тесты, чтобы
не потерять их и ловить регрессии при дальнейших изменениях.

Запуск:
    pip install pytest --break-system-packages
    pytest test_llm_api.py -v
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_api import LLMClient, _extract_error_message, _SoftAPIError


# ---------------------------------------------------------------------
# _extract_error_message: детект "мягкой" ошибки (HTTP 200, но
# {"error": ...} в теле) - реальный случай: OpenRouter/Nvidia вернул
# ResourceExhausted вместо choices, раньше это падало с невнятным
# KeyError: 'choices'.
# ---------------------------------------------------------------------

def test_extract_error_message_detects_error():
    bad_raw = {"error": {"message": "Rate limit exceeded for free model", "code": 429}}
    assert _extract_error_message(bad_raw) == "Rate limit exceeded for free model"


def test_extract_error_message_none_on_normal_response():
    good_raw = {"choices": [{"message": {"content": "ok"}}]}
    assert _extract_error_message(good_raw) is None


# ---------------------------------------------------------------------
# Retry на "мягкую" ошибку - раньше такие ошибки сразу поднимались
# наружу без единой попытки повтора (в отличие от HTTP-уровневых 429/5xx,
# для которых retry уже был). Реальный случай:
# "ResourceExhausted: Worker local total request limit reached (36/32)"
# ---------------------------------------------------------------------

def test_soft_error_retries_then_succeeds():
    client = LLMClient(
        base_url="https://openrouter.ai/api/v1", api_key="test", model="test-model",
        api_format="openai", error_retries=2, error_retry_wait_sec=0,
    )
    call_count = {"n": 0}

    def fake_post_via_urllib(self, url, payload, headers, timeout):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return {"error": {"message": "ResourceExhausted: temp overload"}}
        return {"choices": [{"message": {"content": "OK answer after retries"}}]}

    with patch.object(LLMClient, "_post_via_urllib", fake_post_via_urllib):
        result = client.chat(system="s", user="u")

    assert result == "OK answer after retries"
    assert call_count["n"] == 3  # 2 неудачные попытки + 1 успешная


def test_soft_error_exhausts_retries_and_raises():
    client = LLMClient(
        base_url="https://openrouter.ai/api/v1", api_key="test", model="test-model",
        api_format="openai", error_retries=2, error_retry_wait_sec=0,
    )

    def always_fails(self, url, payload, headers, timeout):
        return {"error": {"message": "ResourceExhausted: still overloaded"}}

    with patch.object(LLMClient, "_post_via_urllib", always_fails):
        try:
            client.chat(system="s", user="u")
            assert False, "должно было выброситься исключение"
        except RuntimeError as e:
            assert "still overloaded" in str(e)
            assert "3 попыток" in str(e)  # error_retries=2 -> всего 3 попытки


# ---------------------------------------------------------------------
# Докрутка обрезанного ответа (finish_reason == "length") - реальный
# случай: ответ обрывался на полуслове ("...задержи"), валидатор
# справедливо называл это "битой генерацией", хотя проблема была
# только в лимите max_tokens, а не в содержании.
# ---------------------------------------------------------------------

def test_truncated_response_gets_continued():
    client = LLMClient(
        base_url="https://openrouter.ai/api/v1", api_key="x", model="test-model",
        api_format="openai",
    )
    call_log = []

    def fake_post(self, url, payload, timeout=None):
        call_log.append(payload["messages"])
        if len(call_log) == 1:
            return {
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": "СОДЕРЖАТЕЛЬНЫЙ БЛОК\nВ аэропорту Казани 7 августа задержи"},
                }]
            }
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "ваются 50 рейсов из-за атаки БПЛА."},
            }]
        }

    with patch.object(LLMClient, "_post", fake_post):
        result = client.chat(system="система", user="вопрос про рейсы")

    assert result == "СОДЕРЖАТЕЛЬНЫЙ БЛОК\nВ аэропорту Казани 7 августа задерживаются 50 рейсов из-за атаки БПЛА."
    assert len(call_log) == 2
    # Второй вызов должен содержать историю диалога с обрубленным ответом
    assert call_log[1][2]["role"] == "assistant"
    assert "задержи" in call_log[1][2]["content"]


def test_no_continuation_when_not_truncated():
    client = LLMClient(
        base_url="https://openrouter.ai/api/v1", api_key="x", model="test-model",
        api_format="openai",
    )
    call_count = {"n": 0}

    def fake_post_ok(self, url, payload, timeout=None):
        call_count["n"] += 1
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Полный ответ без обрывов."}}]}

    with patch.object(LLMClient, "_post", fake_post_ok):
        result = client.chat(system="s", user="u")

    assert result == "Полный ответ без обрывов."
    assert call_count["n"] == 1  # без обрыва - ровно один вызов, без лишних затрат


def test_continuation_stops_at_max_continuations():
    client = LLMClient(
        base_url="https://openrouter.ai/api/v1", api_key="x", model="test-model",
        api_format="openai",
    )
    call_count = {"n": 0}

    def fake_post_always_truncated(self, url, payload, timeout=None):
        call_count["n"] += 1
        return {"choices": [{"finish_reason": "length",
                             "message": {"content": f"часть{call_count['n']} "}}]}

    with patch.object(LLMClient, "_post", fake_post_always_truncated):
        result = client.chat(system="s", user="u", max_continuations=2)

    assert result == "часть1часть2часть3"
    assert call_count["n"] == 3  # max_continuations=2 -> максимум 3 попытки, не бесконечный цикл


# ---------------------------------------------------------------------
# Бюджет контекста: num_ctx (ollama) vs context_budget_tokens (remote) -
# удалённые провайдеры не имеют num_ctx, без явной настройки не могли
# понять свой реальный лимит контекста.
# ---------------------------------------------------------------------

def test_input_char_budget_from_num_ctx():
    client = LLMClient(base_url="http://localhost:11434", api_key="x", model="m",
                       api_format="ollama", num_ctx=8192)
    assert client.input_char_budget > 0
    assert client.input_char_budget < 8192 * 4  # с учётом резерва под системный промпт


def test_input_char_budget_from_context_budget_tokens():
    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="x", model="m",
                       api_format="openai", context_budget_tokens=4000)
    assert client.input_char_budget > 0


def test_input_char_budget_default_fallback():
    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="x", model="m",
                       api_format="openai")
    # без num_ctx и без context_budget_tokens - используется дефолт 8000 токенов
    assert client.input_char_budget > 0


# ---------------------------------------------------------------------
# Прокси: локальный хост (Ollama) НИКОГДА не идёт через прокси, даже
# если proxy_enabled=True - иначе локальный Ollama становится
# недоступен через SOCKS5, который настроен на выход в интернет.
# ---------------------------------------------------------------------

def test_local_host_never_uses_proxy():
    client = LLMClient(base_url="http://localhost:11434", api_key="ollama", model="test",
                       api_format="ollama", proxy_enabled=True)
    assert client._is_local_host is True
    use_proxy = client.proxy_enabled and not client._is_local_host
    assert use_proxy is False


def test_remote_host_uses_proxy_when_enabled():
    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="x", model="m",
                       api_format="openai", proxy_enabled=True)
    assert client._is_local_host is False
    use_proxy = client.proxy_enabled and not client._is_local_host
    assert use_proxy is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
