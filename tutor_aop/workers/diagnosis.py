"""Diagnosis worker: identifies the student's first error step and type."""
from __future__ import annotations

from ..utils import safe_json_loads, render_dialogue
from ..prompts.worker_prompts import DIAGNOSIS_SYSTEM, DIAGNOSIS_USER


DEFAULT_DIAGNOSIS = {
    "first_error_step": "unknown",
    "error_type": "other",
    "misconception": "none",
    "prerequisite_gap": "none",
}


class DiagnosisWorker:
    name = "diagnosis"

    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 512):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, problem: str, initial_solution: str, dialogue: list, subtask: str = "") -> dict:
        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM},
            {"role": "user", "content": DIAGNOSIS_USER.format(
                problem=problem,
                initial_solution=initial_solution or "(none)",
                dialogue=render_dialogue(dialogue) or "(empty)",
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        if not isinstance(parsed, dict):
            parsed = dict(DEFAULT_DIAGNOSIS)
        for k, v in DEFAULT_DIAGNOSIS.items():
            parsed.setdefault(k, v)
        parsed["_raw"] = raw
        return parsed
