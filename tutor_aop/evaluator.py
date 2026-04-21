"""Answer evaluator for Math-500 (and similar \\boxed{}-format datasets)."""
from __future__ import annotations

from .utils import extract_boxed, answers_equivalent


def grade(predicted_text: str, gold_answer: str) -> dict:
    pred_boxed = extract_boxed(predicted_text)
    if pred_boxed is None:
        return {"correct": False, "predicted_boxed": None, "gold": gold_answer, "reason": "no \\boxed{} found"}
    correct = answers_equivalent(pred_boxed, gold_answer)
    return {
        "correct": bool(correct),
        "predicted_boxed": pred_boxed,
        "gold": gold_answer,
        "reason": "ok" if correct else "mismatch",
    }
