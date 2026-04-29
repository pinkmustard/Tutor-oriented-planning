"""Student simulator: initial solve, multi-turn response, independent post-hoc re-solve.

Dialogue-format contract (mirroring PedagogicalRL's ``classroom.py``):

* ``initial_solve`` is a single-shot call with its own system prompt.
* ``respond`` and ``independent_resolve`` both use the SAME system prompt
  (``STUDENT_DIALOGUE_SYSTEM``, problem embedded) and pass the whole dialogue
  as perspective-rotated chat messages -- student utterances become
  ``assistant`` turns, tutor utterances become ``user`` turns. The tutor's
  latest utterance is already the last ``user`` turn.
* Both ``respond`` and ``independent_resolve`` then append a final ``user``
  nudge instructing the model what to produce on this turn:
    - ``respond``:           ``STUDENT_RESPOND_NUDGE`` -- "now write your short reply"
    - ``independent_resolve``: ``STUDENT_FINAL_USER``   -- "now write the full solution"
  This mirrors the explicit "now do X" pattern used by initial_solve and
  keeps brevity constraints visible at the end of the prompt where the model
  attends most strongly.
"""
from __future__ import annotations

from typing import List, Dict

from .prompts.student_prompts import (
    STUDENT_INITIAL_SYSTEM,
    STUDENT_INITIAL_USER,
    STUDENT_DIALOGUE_SYSTEM,
    STUDENT_RESPOND_NUDGE,
    STUDENT_FINAL_USER,
)
from .utils import strip_thinking


def _dialogue_as_student_chat(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Perspective-rotate a neutral dialogue for the student model.

    Student utterances -> ``assistant``; tutor utterances -> ``user``.
    ``<think>...</think>`` blocks are stripped from tutor utterances so the
    student model never sees the tutor's hidden reasoning (PedagogicalRL
    convention). The strip is a no-op for tutors that don't emit ``<think>``,
    so it's safe to apply unconditionally. Unknown roles are silently skipped.
    """
    msgs: List[Dict[str, str]] = []
    for d in dialogue:
        role = d.get("role")
        content = d.get("content", "")
        if role == "student":
            msgs.append({"role": "assistant", "content": content})
        elif role == "tutor":
            msgs.append({"role": "user", "content": strip_thinking(content)})
    return msgs


class StudentAgent:
    def __init__(
        self,
        client,
        temperature: float = 0.6,
        initial_max_tokens: int = 1024,
        respond_max_tokens: int = 320,
        resolve_max_tokens: int = 1024,
    ):
        self.client = client
        self.temperature = temperature
        self.initial_max_tokens = initial_max_tokens
        self.respond_max_tokens = respond_max_tokens
        self.resolve_max_tokens = resolve_max_tokens

    def initial_solve(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": STUDENT_INITIAL_SYSTEM},
            {"role": "user", "content": STUDENT_INITIAL_USER.format(problem=problem)},
        ]
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.initial_max_tokens,
        ).strip()

    def respond(self, problem: str, dialogue: List[Dict[str, str]]) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STUDENT_DIALOGUE_SYSTEM.format(problem=problem)},
        ]
        messages.extend(_dialogue_as_student_chat(dialogue))
        messages.append({"role": "user", "content": STUDENT_RESPOND_NUDGE})
        return self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.respond_max_tokens,
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
