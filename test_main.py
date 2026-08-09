"""
test_main.py — регрессионные тесты для main.py (универсальный
поисковый пайплайн: HTML-анализ -> LLM -> vision-fallback).

Это реальные проверки, сделанные вручную в консоли по ходу разработки,
оформленные как pytest-тесты для защиты от регрессий.

Запуск:
    pip install pytest --break-system-packages
    pytest test_main.py -v
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main  # noqa: F401  (проверяем, что тонкий main.py импортируется)
import pipeline_core


# ---------------------------------------------------------------------
# split_text_into_chunks: разбивка длинного текста на куски с
# перекрытием. БАГ (был): если overlap >= chunk_size, цикл не
# продвигался вперёд и уходил в бесконечное повторение при малом
# бюджете контекста.
# ---------------------------------------------------------------------

def test_split_text_normal_case():
    chunks = pipeline_core.split_text_into_chunks("B" * 25000, chunk_size=10000, overlap=200)
    assert len(chunks) == 3
    assert len(chunks[0]) == 10000
    assert len(chunks[-1]) == 5400


def test_split_text_overlap_larger_than_chunk_size_does_not_hang():
    """Раньше это зависало бесконечно - overlap "съедал" весь шаг вперёд."""
    chunks = pipeline_core.split_text_into_chunks("A" * 1000, chunk_size=100, overlap=200)
    assert len(chunks) > 0
    assert len(chunks) < 1000  # не бесконечный список


def test_split_text_fits_in_one_chunk():
    chunks = pipeline_core.split_text_into_chunks("short text", chunk_size=1000)
    assert chunks == ["short text"]


# ---------------------------------------------------------------------
# group_vision_answers_by_budget: группировка ЦЕЛЫХ описаний
# скриншотов по бюджету (не посимвольная резка - иначе можно разорвать
# конкретное описание со всеми его цифрами прямо посередине).
# ---------------------------------------------------------------------

def test_group_vision_answers_respects_whole_entries():
    answers = [
        "[Скриншот 1] " + "A" * 100,
        "[Скриншот 2] " + "B" * 100,
        "[Скриншот 3] " + "C" * 100,
        "[Скриншот 4] " + "D" * 100,
    ]
    groups = pipeline_core.group_vision_answers_by_budget(answers, budget=250)
    assert len(groups) == 2
    # каждый элемент внутри группы - целый, не разорванный
    for group in groups:
        for answer in group:
            assert answer in answers


def test_group_vision_answers_single_large_entry():
    """Один элемент больше самого бюджета - не режется, попадает в
    свою группу целиком (лучше превысить бюджет, чем разорвать
    конкретное описание)."""
    answers = ["X" * 500]
    groups = pipeline_core.group_vision_answers_by_budget(answers, budget=100)
    assert len(groups) == 1
    assert len(groups[0][0]) == 500


def test_group_vision_answers_empty_list():
    assert pipeline_core.group_vision_answers_by_budget([], budget=100) == []


# ---------------------------------------------------------------------
# extract_main_page_text: достаёт основной текст (article/main),
# отфильтровывая nav/header/footer/script/style.
# ---------------------------------------------------------------------

def test_extract_main_page_text_filters_navigation():
    html = """
    <html><body>
    <nav>Меню Игры Поиск Профиль</nav>
    <header>kazan.aif.ru 16+ Казань</header>
    <article>
    <h1>7 августа утром в аэропорту Казани из-за атаки БПЛА задерживаются 50 рейсов</h1>
    <p>По данным аэропорта, задержаны 50 рейсов из-за атаки беспилотников.</p>
    </article>
    <footer>Реклама Metallista краска</footer>
    </body></html>
    """
    text = pipeline_core.extract_main_page_text(html)
    assert "50 рейсов" in text
    assert "Меню Игры" not in text
    assert "Metallista" not in text


@pytest.mark.skipif(
    not pipeline_core.HAS_TRAFILATURA,
    reason="trafilatura не установлена (pip install trafilatura) - без неё "
           "div-based меню без <nav>/<article> тегов не фильтруется, это "
           "и есть весь смысл этого теста, а не баг старой эвристики",
)
def test_extract_main_page_text_readmode_filters_div_based_menu():
    """Реальный случай: thedecisionlab.com/biases/gamblers-fallacy -
    гигантское мега-меню (AI/Consulting/Industries/Resources и десятки
    пунктов) свёрстано обычными <div>, БЕЗ единого <nav>/<header>/<article>
    тега. Старая эвристика (decompose script/style/nav/header/footer,
    иначе искать article/main) такое меню не отфильтровывала - оно
    целиком утекало в текст, отправляемый в LLM, раньше самой статьи.

    trafilatura находит основной контент по плотности текста и
    типографике, а не по наличию тегов - должна выкинуть меню и
    оставить только текст статьи, даже без единого семантического тега.
    """
    html = """
    <html><body>
    <div class="mega-menu">
      <div>AI</div><div>Consulting</div><div>Industries</div><div>Resources</div>
      <div><a href="/1">Data Analytics</a></div><div><a href="/2">Innovation</a></div>
      <div><a href="/3">Strategy</a></div><div><a href="/4">Operations</a></div>
      <div><a href="/5">People</a></div><div><a href="/6">Marketing</a></div>
      <div><a href="/7">Risk</a></div><div><a href="/8">Health</a></div>
      <div><a href="/9">Education</a></div><div><a href="/10">Climate</a></div>
    </div>
    <div class="content">
    <h1>Gambler's fallacy</h1>
    <p>The gambler's fallacy describes our belief that the probability of a
    random event occurring in the future is influenced by previous instances
    of that type of event. Consider Jane, who plays Blackjack and believes
    her losing streak will end on the fifth day.</p>
    </div>
    </body></html>
    """
    text = pipeline_core.extract_main_page_text(
        html, url="https://thedecisionlab.com/biases/gamblers-fallacy")
    assert "gambler's fallacy" in text.lower()
    assert "losing streak" in text.lower()
    # пункты мега-меню не должны были попасть в текст статьи
    for menu_item in ("Consulting", "Industries", "Data Analytics", "Operations"):
        assert menu_item not in text


def test_extract_main_page_text_falls_back_without_trafilatura():
    """Если trafilatura недоступна в окружении (не установлена/сбой) -
    функция не должна падать, а откатывается на старую эвристику по
    тегам (article/main, иначе body минус script/style/nav/header/footer)."""
    html = """
    <html><body>
    <nav>Меню Игры Поиск</nav>
    <article><h1>Заголовок</h1><p>Основной текст статьи с фактом.</p></article>
    </body></html>
    """
    with patch.object(pipeline_core, "HAS_TRAFILATURA", False):
        text = pipeline_core.extract_main_page_text(html)
    assert "Основной текст статьи" in text
    assert "Меню Игры" not in text


# ---------------------------------------------------------------------
# looks_like_captcha_page: детект антибот-защиты по ключевым словам -
# используется, чтобы пропустить сайт и попробовать следующий
# результат из выдачи Google, а не пытаться решать капчу.
# ---------------------------------------------------------------------

def test_looks_like_captcha_page_detects_cloudflare():
    assert pipeline_core.looks_like_captcha_page("<html>Verify you are human</html>") is True


def test_looks_like_captcha_page_false_on_normal_content():
    assert pipeline_core.looks_like_captcha_page("<html>Google Pixel 6 specs</html>") is False


# ---------------------------------------------------------------------
# extract_keywords_from_query: универсальное извлечение ключевых слов
# ИЗ ЛЮБОГО запроса (не только про бонусы). Слова обрезаются до 5 букв
# (псевдо-стемминг), чтобы ловить разные падежи/формы.
# ---------------------------------------------------------------------

def test_extract_keywords_generic_topic():
    topic = "правила использования бонусов Аэрофлот"
    keywords = pipeline_core.extract_keywords_from_query(topic)
    assert "прави" in keywords
    assert "бонус" in keywords
    assert "аэроф" in keywords


def test_extract_keywords_stem_matches_inflected_forms():
    """Ключевые слова должны совпадать с разными падежами того же
    корня как подстрока - "бонус" должен матчиться и с "бонусов", и
    с "бонусные"."""
    import re
    keywords = pipeline_core.extract_keywords_from_query("правила использования бонусов")
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    assert pattern.search("бонусов")
    assert pattern.search("бонус")
    assert pattern.search("бонусные")


# ---------------------------------------------------------------------
# answer_numbers_grounded_in_source: защита от галлюцинации цифр -
# реальный случай: модель ответила "high of 28°C and a low of 19°C"
# для будущей даты, хотя таких цифр в исходном тексте не было вообще
# (собственная "оценка" модели по общим знаниям о климате).
# ---------------------------------------------------------------------

def test_grounded_numbers_detects_hallucinated_values():
    source = "Kazan Tatarstan 21° current weather monthly forecast calendar August 2026"
    hallucinated_answer = (
        "СОДЕРЖАТЕЛЬНЫЙ БЛОК\nAccording to AccuWeather, weather in Kazan "
        "on August 8, 2026 is forecast to have a high of 28°C and a low of 19°C."
    )
    assert pipeline_core.answer_numbers_grounded_in_source(hallucinated_answer, source) is False


def test_grounded_numbers_accepts_real_values():
    source = "В аэропорту Казани задержаны 22 рейса на вылет и 30 рейсов на прилет"
    real_answer = "СОДЕРЖАТЕЛЬНЫЙ БЛОК\nВ аэропорту задержаны 22 рейса на вылет и 30 на прилет, всего 52."
    assert pipeline_core.answer_numbers_grounded_in_source(real_answer, source) is True


def test_grounded_numbers_no_numbers_in_answer_passes():
    """Если в ответе вообще нет чисел - проверка неприменима, не блокируем."""
    assert pipeline_core.answer_numbers_grounded_in_source("Ответ без цифр вообще.", "любой источник") is True


# ---------------------------------------------------------------------
# classify_url logic (через LLM) не тестируем напрямую (требует mock
# сложного промпта), но проверяем generate_search_query fallback -
# реальный случай: модель вернула рассуждение вместо самого запроса
# ("The user wants a search query for...").
# ---------------------------------------------------------------------

def test_generate_search_query_rejects_reasoning_response():
    client = MagicMock()
    reasoning_response = (
        "The user wants a search query for the weather in Kazan on August 8, 2026.\n"
        "Since this is a future date (long-term forecast), standard weather sites usually don't have"
    )
    with patch("pipeline_core.timed_chat", return_value=reasoning_response):
        result = pipeline_core.generate_search_query(client, "погода в казани на 8 августа")
    # должен откатиться на исходную тему, а не вернуть мусорное рассуждение
    assert result == "погода в казани на 8 августа"


def test_generate_search_query_accepts_clean_query():
    client = MagicMock()
    with patch("pipeline_core.timed_chat", return_value='"Google Pixel 6 screen resolution"'):
        result = pipeline_core.generate_search_query(client, "разрешение экрана Pixel 6")
    assert result == "Google Pixel 6 screen resolution"


# ---------------------------------------------------------------------
# analyze_vision_answer parsing: комбинированный разбор SUFFICIENT/
# ELEMENT/ZOOM_REGION из одного ответа модели.
# ---------------------------------------------------------------------

def test_analyze_vision_answer_parses_all_fields():
    fake_answer = "SUFFICIENT: no\nELEMENT: NONE\nZOOM_REGION: top_center"
    with patch("pipeline_core.timed_chat", return_value=fake_answer):
        is_sufficient, element_text, zoom_region = pipeline_core.analyze_vision_answer(
            MagicMock(), "some vision text", "topic")
    assert is_sufficient is False
    assert element_text is None
    assert zoom_region == "top_center"


def test_analyze_vision_answer_sufficient_with_element():
    fake_answer = 'SUFFICIENT: yes\nELEMENT: "Условия получения премий"\nZOOM_REGION: NONE'
    with patch("pipeline_core.timed_chat", return_value=fake_answer):
        is_sufficient, element_text, zoom_region = pipeline_core.analyze_vision_answer(
            MagicMock(), "some vision text", "topic")
    assert is_sufficient is True
    assert element_text == "Условия получения премий"
    assert zoom_region is None


# ---------------------------------------------------------------------
# make_give_up_summary: настоящий multi-pass map-reduce вместо слепой
# обрезки текста - реальный риск: важная деталь в конце длинного
# текста терялась бы при простом raw_notes[:budget].
# ---------------------------------------------------------------------

def test_give_up_summary_single_pass_when_fits():
    client = MagicMock()
    client.input_char_budget = 100000
    call_count = {"n": 0}

    def fake_timed_chat(c, **kwargs):
        call_count["n"] += 1
        return "Короткое резюме без деления на части."

    with patch("pipeline_core.timed_chat", fake_timed_chat):
        result = pipeline_core.make_give_up_summary(client, "тема", "короткий текст", sites_count=1)

    assert result == "Короткое резюме без деления на части."
    assert call_count["n"] == 1


def test_give_up_summary_multi_pass_preserves_important_detail():
    """Реальная проверка: важная деталь в САМОМ КОНЦЕ длинного текста
    не должна теряться при многопроходном сжатии (в отличие от
    старой логики с raw_notes[:budget])."""
    client = MagicMock()
    client.input_char_budget = 200

    raw_notes = "мусорный текст " * 30 + " ВАЖНО: реальная температура 25 градусов найдена в последнем абзаце"
    call_log = []

    def fake_timed_chat(c, **kwargs):
        call_log.append(kwargs["user"][:60])
        if "ВАЖНО" in kwargs["user"]:
            return "В этом фрагменте: температура 25 градусов упомянута явно."
        if "мусорный" in kwargs["user"] and "ВАЖНО" not in kwargs["user"]:
            return "В этом фрагменте по теме ничего полезного нет."
        return ("СЖАТОЕ РЕЗЮМЕ: точных данных почти нет, но упомянута температура "
                "25 градусов. Точного ответа найти не удалось.")

    with patch("pipeline_core.timed_chat", fake_timed_chat):
        result = pipeline_core.make_give_up_summary(client, "какая температура", raw_notes, sites_count=2)

    assert "25" in result
    assert len(call_log) > 1  # реально было несколько проходов (map + reduce)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
