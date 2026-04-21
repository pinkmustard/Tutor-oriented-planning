"""Pedagogical Utterance Auditor: LLM-judge on the drafted tutor utterance."""
from __future__ import annotations

from .utils import safe_json_loads
from .prompts.auditor_prompts import AUDITOR_SYSTEM, AUDITOR_USER


DEFAULT_AUDIT_OK = {
    "pedagogically_compliant": True,
    "answer_leaked": False,
    "socratic_style": True,
    "length_ok": True,
    "reasons": [],
    "suggestions": "",
}


class PedagogicalAuditor:
    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 384):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def audit(self, problem: str, draft: str, tutor_move: str = "") -> dict:
        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM},
            {"role": "user", "content": AUDITOR_USER.format(
                problem=problem,
                tutor_move=tutor_move or "(unknown)",
                draft=draft,
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        if not isinstance(parsed, dict):
            parsed = dict(DEFAULT_AUDIT_OK)
        for k, v in DEFAULT_AUDIT_OK.items():
            parsed.setdefault(k, v)
        parsed["_raw"] = raw
        return parsed

    @staticmethod
    def needs_revision(audit_output: dict) -> bool:
        return not bool(audit_output.get("pedagogically_compliant", True))
