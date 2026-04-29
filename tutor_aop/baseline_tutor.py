"""Baseline tutor: vanilla tutor-student dialogue with no planning layer.

One LLM call per turn. The system prompt carries only the problem; the
student's initial attempt is the first ``user`` message in the dialogue chat
(student -> "user", tutor -> "assistant"). After the perspective-rotated
dialogue we append an explicit user-role nudge instructing the model to
produce its next short utterance -- mirrors the "now do X" pattern that
initial_solve / independent_resolve already use, and keeps the brevity
constraint visible at the end of the prompt where attention concentrates.
"""
from __future__ import annotations

from typing import List, Dict

from .prompts.baseline_tutor_prompts import (
    BASELINE_TUTOR_SYSTEM,
    BASELINE_TUTOR_RESPOND_NUDGE,
)


class BaselineTutor:
    def __init__(
        self,
        client,
        system_prompt: str = BASELINE_TUTOR_SYSTEM,
        respond_nudge: str = BASELINE_TUTOR_RESPOND_NUDGE,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.respond_nudge = respond_nudge
        self.temperature = temperature
        self.max_tokens = max_tokens

    def respond(
        self,
        problem: str,
        dialogue: List[Dict[str, str]],
    ) -> str:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt.format(problem=problem),
            }
        ]
        # Student utterances become "user"; tutor utterances become "assistant".
        for d in dialogue:
            role = d.get("role")
            content = d.get("content", "")
            if role == "student":
                messages.append({"role": "user", "content": content})
            elif role == "tutor":
                messages.append({"role": "assistant", "content": content})
        # Final user-role nudge -- "now write your next utterance"
        messages.append({"role": "user", "content": self.respond_nudge})
        out = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (out or "").strip()
