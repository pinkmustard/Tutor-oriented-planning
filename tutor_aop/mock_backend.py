"""Deterministic mock backend for --mock dry runs.

Returns structurally-valid responses for every component so the full
pipeline can be exercised without a live vLLM server.
"""
from __future__ import annotations

import json


def _join(messages):
    return "\n".join(m.get("content", "") for m in messages)


def default_mock_handler(model: str, messages, temperature, max_tokens, stop):
    text = _join(messages).lower()

    if "plan detector" in text or "completeness and non-redundancy" in text:
        return json.dumps({
            "satisfies_completeness": True,
            "satisfies_non_redundancy": True,
            "issues": [],
            "suggestions": "",
        })

    if "pedagogical utterance auditor" in text or "answer leakage" in text:
        return json.dumps({
            "pedagogically_compliant": True,
            "answer_leaked": False,
            "socratic_style": True,
            "reasons": [],
            "suggestions": "",
        })

    if "diagnosis worker" in text:
        return json.dumps({
            "first_error_step": "Step not identified in mock mode.",
            "error_type": "procedural",
            "misconception": "none detected in mock",
            "prerequisite_gap": "none",
        })

    if "tutor move selector" in text:
        return json.dumps({
            "selected_move": "Probing",
            "rationale": "Mock: probe the student's reasoning.",
        })

    if "search query" in text and "retrieval" in text:
        return "pythagorean theorem"

    if "meta-tutor" in text and "agenda" in text:
        return json.dumps({
            "agenda": [
                {"id": 1, "task": "Diagnose the student's most recent error",
                 "worker": "diagnosis", "reason": "Identify error", "dep": []},
                {"id": 2, "task": "Select an appropriate tutor move",
                 "worker": "tutor_move", "reason": "Decide pedagogy", "dep": [1]},
            ]
        })

    if "final tutor response" in text or "generate the tutor's next utterance" in text:
        return "Let's take another look at your work. Can you walk me through how you got the last step?"

    if "final answer" in text.lower() and "boxed" in text:
        return "After reviewing, the answer is \\boxed{42}."

    if "conversation with the tutor has now ended" in text:
        return "Reviewing the conversation... Step 1: ... The answer is \\boxed{42}."

    if "solve it yourself" in text or "on your own" in text:
        return "I'll try. ... The answer is \\boxed{0}."

    if "you are a student in a conversation" in text:
        return "I think I see. Let me try that step again. I get 2x - 3 = 0, so x = 3/2."

    if "you are a student" in text:
        return "I'll try. ... The answer is \\boxed{0}."

    return "OK"
