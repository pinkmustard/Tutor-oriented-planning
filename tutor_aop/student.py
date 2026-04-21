"""Student simulator: initial solve, multi-turn response, independent post-hoc re-solve."""
from __future__ import annotations

from .utils import render_dialogue
from .prompts.student_prompts import (
    STUDENT_INITIAL_SYSTEM,
    STUDENT_INITIAL_USER,
    STUDENT_RESPOND_SYSTEM,
    STUDENT_RESPOND_USER,
    STUDENT_RESOLVE_SYSTEM,
    STUDENT_RESOLVE_USER,
)


class StudentAgent:
    def __init__(self, client, temperature: float = 0.7, max_tokens: int = 1024):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def initial_solve(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": STUDENT_INITIAL_SYSTEM},
            {"role": "user", "content": STUDENT_INITIAL_USER.format(problem=problem)},
        ]
        return self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens).strip()

    def respond(
        self,
        problem: str,
        initial_solution: str,
        dialogue: list,
        tutor_utterance: str,
    ) -> str:
        messages = [
            {"role": "system", "content": STUDENT_RESPOND_SYSTEM},
            {"role": "user", "content": STUDENT_RESPOND_USER.format(
                problem=problem,
                initial_solution=initial_solution,
                dialogue=render_dialogue(dialogue) or "(empty)",
                tutor_utterance=tutor_utterance,
            )},
        ]
        return self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens).strip()

    def independent_resolve(self, problem: str, dialogue: list) -> str:
        messages = [
            {"role": "system", "content": STUDENT_RESOLVE_SYSTEM},
            {"role": "user", "content": STUDENT_RESOLVE_USER.format(
                problem=problem,
                dialogue=render_dialogue(dialogue) or "(empty)",
            )},
        ]
        return self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens).strip()
