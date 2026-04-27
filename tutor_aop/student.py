"""Student simulator: initial solve, multi-turn response, independent post-hoc re-solve.

Dialogue-format contract (mirroring PedagogicalRL's ``classroom.py``):

* ``initial_solve`` is a single-shot call with its own system prompt.
* ``respond`` and ``independent_resolve`` both use the SAME system prompt
  (``STUDENT_DIALOGUE_SYSTEM``, problem embedded) and pass the whole dialogue
  as perspective-rotated chat messages -- student utterances become
  ``assistant`` turns, tutor utterances become ``user`` turns. The tutor's
  latest utterance is already the last ``user`` turn, so there is no separate
  ``tutor_utterance`` input.
* ``independent_resolve`` further appends a single ``user`` turn
  (``STUDENT_FINAL_USER``) asking the student to produce the full solution.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from .prompts.student_prompts import (
    STUDENT_INITIAL_SYSTEM,
    STUDENT_INITIAL_USER,
    STUDENT_DIALOGUE_SYSTEM,
    STUDENT_FINAL_USER,
)


def _dialogue_as_student_chat(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Perspective-rotate a neutral dialogue for the student model.

    Student utterances -> ``assistant``; tutor utterances -> ``user``.
    Unknown roles are silently skipped.
    """
    msgs: List[Dict[str, str]] = []
    for d in dialogue:
        role = d.get("role")
        content = d.get("content", "")
        if role == "student":
            msgs.append({"role": "assistant", "content": content})
        elif role == "tutor":
            msgs.append({"role": "user", "content": content})
    return msgs


class StudentAgent:
    def __init__(
        self,
        client,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        resolve_max_tokens: Optional[int] = None,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        # independent_resolve needs room for a full solution after a long
        # dialogue, so let callers allocate extra tokens just for that step.
        self.resolve_max_tokens = resolve_max_tokens or max_tokens

    def initial_solve(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": STUDENT_INITIAL_SYSTEM},
            {"role": "user", "content": STUDENT_INITIAL_USER.format(problem=problem)},
        ]
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()

    def respond(self, problem: str, dialogue: List[Dict[str, str]]) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STUDENT_DIALOGUE_SYSTEM.format(problem=problem)},
        ]
        messages.extend(_dialogue_as_student_chat(dialogue))
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()

    def independent_resolve(self, problem: str, dialogue: List[Dict[str, str]]) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STUDENT_DIALOGUE_SYSTEM.format(problem=problem)},
        ]
        messages.extend(_dialogue_as_student_chat(dialogue))
        messages.append({"role": "user", "content": STUDENT_FINAL_USER})
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.resolve_max_tokens,
        ).strip()
