"""Tutor Move Selector worker: picks one of Focus / Probing / Telling / Generic."""
from __future__ import annotations

import json

from ..utils import safe_json_loads, render_dialogue
from ..prompts.worker_prompts import TUTOR_MOVE_SYSTEM, TUTOR_MOVE_USER


VALID_MOVES = {"Focus", "Probing", "Telling", "Generic"}
DEFAULT_MOVE = {"selected_move": "Probing", "rationale": "fallback default"}


class TutorMoveWorker:
    name = "tutor_move"

    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 256):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, problem: str, diagnosis: dict, dialogue: list, subtask: str = "") -> dict:
        diag_str = json.dumps(diagnosis, ensure_ascii=False) if diagnosis else "(none)"
        messages = [
            {"role": "system", "content": TUTOR_MOVE_SYSTEM},
            {"role": "user", "content": TUTOR_MOVE_USER.format(
                problem=problem,
                diagnosis=diag_str,
                dialogue=render_dialogue(dialogue) or "(empty)",
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        if not isinstance(parsed, dict) or parsed.get("selected_move") not in VALID_MOVES:
            parsed = dict(DEFAULT_MOVE)
        parsed.setdefault("rationale", "")
        parsed["_raw"] = raw
        return parsed
