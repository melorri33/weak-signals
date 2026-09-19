"""Конвейер «запрос → топ-15» (Интеграция).

Импорт:
    from src.pipeline.run import run                    # весь конвейер
    from src.pipeline.candidates import extract_candidates

Функцию run здесь намеренно не пере-экспортируем: имя совпало бы с модулем src.pipeline.run,
и тогда `src.pipeline.run` означало бы функцию, а не модуль — это ломает подмену в тестах.
"""
