"""Meta-Tutor: agenda generation, replan, final response, revision.

Design choices:

* Planning calls (``plan_agenda``, ``replan``) emit JSON. Their input is a
  flat-text rendered dialogue inside a single user message -- fine for JSON
  output workloads, and the prompts are tight.

* The actual student-facing utterance generation (``generate_final``,
  ``revise_final``) uses **proper multi-turn chat** with perspective rotation
  (student -> ``user``, tutor -> ``assistant``), mirroring baseline_tutor.
  Older Qwen-family checkpoints degrade noticeably when long prior dialogue
  is dumped as flat text inside one user message; running the same content
  as natural chat turns keeps the chat template happy and prevents the model
  from drifting into "let me solve the whole problem" mode.

* Worker findings (diagnosis / tutor_move) are summarised into a compact
  label-form string and included only inside the final user-role nudge --
  not as a JSON dump in the dialogue context -- so verbose LaTeX-quoted
  diagnosis fields can't leak into the visible reply.

* ``generate_final`` / ``revise_final`` use ``tutor_turn_max_tokens`` (small,
  matches baseline). All other meta-tutor calls keep ``max_tokens`` because
  they emit JSON and need headroom.
"""
from __future__ import annotations

import json
from typing import List, Dict

from .utils import safe_json_loads, render_dialogue
from .prompts.meta_tutor_prompts import (
    META_TUTOR_AGENDA_SYSTEM,
    META_TUTOR_AGENDA_USER,
    META_TUTOR_REPLAN_SYSTEM,
    META_TUTOR_REPLAN_USER,
    META_TUTOR_FINAL_SYSTEM,
    META_TUTOR_FINAL_NUDGE,
    META_TUTOR_REVISE_NUDGE,
)


VALID_WORKERS = {"diagnosis", "tutor_move", "retrieval"}
DEFAULT_AGENDA = {
    "agenda": [
        {"id": 1, "task": "Diagnose the student's most recent error",
         "worker": "diagnosis", "reason": "default agenda", "dep": []},
        {"id": 2, "task": "Select an appropriate tutor move",
         "worker": "tutor_move", "reason": "default agenda", "dep": [1]},
    ]
}


def _sanitize_agenda(obj) -> dict:
    """Coerce arbitrary agenda JSON into the expected shape."""
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            obj = {"agenda": obj}
        else:
            return dict(DEFAULT_AGENDA)
    agenda = obj.get("agenda", [])
    if not isinstance(agenda, list):
        return dict(DEFAULT_AGENDA)
    clean = []
    for i, item in enumerate(agenda):
        if not isinstance(item, dict):
            continue
        worker = item.get("worker") or item.get("name")
        if worker not in VALID_WORKERS:
            continue
        clean.append({
            "id": item.get("id", i + 1),
            "task": str(item.get("task", "")).strip() or f"{worker} sub-task",
            "worker": worker,
            "reason": str(item.get("reason", "")).strip(),
            "dep": item.get("dep", []) if isinstance(item.get("dep", []), list) else [],
        })
    if not clean:
        return dict(DEFAULT_AGENDA)
    return {"agenda": clean}


def _dialogue_as_tutor_chat(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Perspective-rotate a neutral dialogue for the tutor model.

    Student utterances -> ``user``; tutor utterances -> ``assistant``.
    Unknown roles are silently skipped.
    """
    msgs: List[Dict[str, str]] = []
    for d in dialogue:
        role = d.get("role")
        content = d.get("content", "")
        if role == "student":
            msgs.append({"role": "user", "content": content})
        elif role == "tutor":
            msgs.append({"role": "assistant", "content": content})
    return msgs


VALID_MOVES = {"Focus", "Probing", "Telling", "Generic"}


def _selected_move(worker_outputs: dict) -> str:
    """Extract just the tutor move name. Diagnosis details / misconception
    strings are intentionally NOT surfaced to ``generate_final`` -- with
    Qwen2.5-7B they prompted the model to demonstrate the diagnosis rather
    than scaffold. The dialogue is enough context to see the student's
    error.
    """
    move = ((worker_outputs.get("tutor_move") or {}).get("selected_move") or "").strip()
    return move if move in VALID_MOVES else "Probing"


class MetaTutor:
    def __init__(
        self,
        client,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tutor_turn_max_tokens: int = 320,
    ):
        self.client = client
        self.temperature = temperature
        # JSON-emitting calls (plan_agenda, replan)
        self.max_tokens = max_tokens
        # Student-facing utterance calls (generate_final, revise_final)
        self.tutor_turn_max_tokens = tutor_turn_max_tokens

    def plan_agenda(
        self,
        problem: str,
        dialogue: list,
        turn_idx: int,
        max_turns: int,
    ) -> dict:
        messages = [
            {"role": "system", "content": META_TUTOR_AGENDA_SYSTEM},
            {"role": "user", "content": META_TUTOR_AGENDA_USER.format(
                problem=problem,
                dialogue=render_dialogue(dialogue) or "(empty)",
                turn_idx=turn_idx,
                max_turns=max_turns,
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        agenda = _sanitize_agenda(parsed)
        agenda["_raw"] = raw
        return agenda

    def replan(
        self,
        problem: str,
        previous_agenda: dict,
        detector_feedback: dict,
        dialogue: list,
    ) -> dict:
        prev = {k: v for k, v in previous_agenda.items() if not k.startswith("_")}
        fb = {k: v for k, v in detector_feedback.items() if not k.startswith("_")}
        messages = [
            {"role": "system", "content": META_TUTOR_REPLAN_SYSTEM},
            {"role": "user", "content": META_TUTOR_REPLAN_USER.format(
                problem=problem,
                previous_agenda=json.dumps(prev, ensure_ascii=False),
                detector_feedback=json.dumps(fb, ensure_ascii=False),
                dialogue=render_dialogue(dialogue) or "(empty)",
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        parsed = safe_json_loads(raw, default=None)
        agenda = _sanitize_agenda(parsed)
        agenda["_raw"] = raw
        return agenda

    def generate_final(
        self,
        problem: str,
        dialogue: list,
        worker_outputs: dict,
        turn_idx: int,
        max_turns: int,
    ) -> str:
        # System carries the role + the problem; dialogue history is passed as
        # real chat messages (perspective rotation); a single final user-role
        # nudge appended to the end carries worker findings + brevity reminder.
        # turn_idx / max_turns are surfaced into the nudge so the model can
        # apply the absolute-termination rule (no <end_of_conversation> on
        # turn 0).
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": META_TUTOR_FINAL_SYSTEM.format(problem=problem)},
        ]
        messages.extend(_dialogue_as_tutor_chat(dialogue))
        messages.append({
            "role": "user",
            "content": META_TUTOR_FINAL_NUDGE.format(
                selected_move=_selected_move(worker_outputs),
                turn_idx=turn_idx,
                max_turns=max_turns,
            ),
        })
        raw = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.tutor_turn_max_tokens,
        )
        return (raw or "").strip()

    def revise_final(
        self,
        problem: str,
        dialogue: list,
        worker_outputs: dict,
        draft: str,
        auditor_feedback: dict,
        turn_idx: int,
        max_turns: int,
    ) -> str:
        # Same system + perspective-rotated dialogue as generate_final.
        # The draft is NOT added to the dialogue (it never reached the
        # student); it goes into the revise nudge.
        fb = {k: v for k, v in auditor_feedback.items() if not k.startswith("_")}
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": META_TUTOR_FINAL_SYSTEM.format(problem=problem)},
        ]
        messages.extend(_dialogue_as_tutor_chat(dialogue))
        messages.append({
            "role": "user",
            "content": META_TUTOR_REVISE_NUDGE.format(
                draft=draft,
                auditor_feedback=json.dumps(fb, ensure_ascii=False),
                selected_move=_selected_move(worker_outputs),
                turn_idx=turn_idx,
                max_turns=max_turns,
            ),
        })
        raw = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.tutor_turn_max_tokens,
        )
        return (raw or "").strip()
