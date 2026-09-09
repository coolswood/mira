"""Playful PR status comments — visible activity for otherwise silent rounds.

Repeat reviews (round 2+) work correctly but quietly: incremental diff,
raised thresholds, and the walkthrough is edited in place, so a PR can go
through several pushes with no visible bot activity. These one-shot
comments restore the feedback loop: one when the review actually starts,
one when it finishes with zero findings. Disable via
``review.chat_updates: false``.
"""

from __future__ import annotations

import random

REVIEW_START_MESSAGES = [
    "🔍 Так-так, что тут у нас… Приступаю к ревью!",
    "☕ Начинаю разбор PR — дифф, покажи, на что ты способен.",
    "🚦 Поехали! Достаю лупу и читаю изменения.",
    "🔬 Ревью стартует: найду — скажу, не найду — тоже скажу.",
    "📦 Разворачиваю дифф и приступаю к разбору.",
    "🏃 Стартую ревью — вернусь с вердиктом.",
    "👀 Пригляделся к изменениям. Начинаю.",
]

REVIEW_CLEAN_MESSAGES = [
    "✅ Готово! Замечаний нет — код чище, чем совесть кота.",
    "🎉 Ревью завершено: всё отлично, можно вливать!",
    "✨ Закончил. Ноль замечаний — красиво.",
    "👌 Всё чисто. Так держать!",
    "🏆 Ревью окончено, проблем не нашёл.",
    "🧼 Разбор закончен — код блестит.",
    "🙆 Придраться не к чему. Закончил, всё отлично!",
]


def review_started_message() -> str:
    return random.choice(REVIEW_START_MESSAGES)


def review_clean_message() -> str:
    return random.choice(REVIEW_CLEAN_MESSAGES)
