"""Plan Detector: plan-level completeness / non-redundancy checker."""
from __future__ import annotations

import json

from .utils import safe_json_loads
from .prompts.detector_prompts import DETECTOR_SYSTEM, DETECTOR_USER


DEFAULT_DETECTOR_OK = {
    "satisfies_completeness": True,
    "satisfies_non_redundancy": True,
    "issues": [],
    "suggestions": "",
}


class PlanDetector:
    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 512):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def detect(self, problem: str, agenda: dict, turn_idx: int, max_turns: int) -> dict:
        agenda_clean = {k: v for k, v in agenda.items() if not k.startswith("_")}
        messages = [
            {"role": "system", "content": DETECTOR_SYSTEM},
            {"role": "user", "content": DETECTOR_USER.format(
                problem=problem,
                turn_idx=turn_idx,
                max_turns=max_turns,
                agenda=json.dumps(agenda_clean, ensure_ascii=False),
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        if not isinstance(parsed, dict):
            parsed = dict(DEFAULT_DETECTOR_OK)
        parsed.setdefault("satisfies_completeness", True)
        parsed.setdefault("satisfies_non_redundancy", True)
        parsed.setdefault("issues", [])
        parsed.setdefault("suggestions", "")
        parsed["_raw"] = raw
        return parsed

    @staticmethod
    def needs_replan(detector_output: dict) -> bool:
        return not (
            bool(detector_output.get("satisfies_completeness", True))
            and bool(detector_output.get("satisfies_non_redundancy", True))
        )
