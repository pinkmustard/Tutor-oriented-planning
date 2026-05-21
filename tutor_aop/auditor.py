"""Pedagogical Utterance Auditor: LLM-judge on the drafted tutor utterance."""
from __future__ import annotations

from .utils import safe_json_loads, render_dialogue
from .prompts.auditor_prompts import AUDITOR_SYSTEM, AUDITOR_USER


DEFAULT_AUDIT_OK = {
    "pedagogically_compliant": True,
    "answer_leaked": False,
    "socratic_style": True,
    "length_ok": True,
    "premature_termination": False,
    "reasons": [],
    "suggestions": "",
}


class PedagogicalAuditor:
    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 384):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def audit(
        self,
        problem: str,
        draft: str,
        tutor_move: str = "",
        dialogue: list | None = None,
    ) -> dict:
        # The dialogue snapshot is required to evaluate the
        # premature_termination criterion (the auditor must see whether
        # the student has had a chance to respond to a prior tutor turn,
        # and whether their most recent message demonstrates a corrected
        # step). Falls back to "(no dialogue available)" only for legacy
        # callers; production runs in aop_classroom always pass it.
        rendered = render_dialogue(dialogue) if dialogue else "(no dialogue available)"
        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM},
            {"role": "user", "content": AUDITOR_USER.format(
                problem=problem,
                tutor_move=tutor_move or "(unknown)",
                dialogue=rendered,
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
