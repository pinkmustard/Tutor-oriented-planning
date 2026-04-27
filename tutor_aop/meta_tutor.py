"""Meta-Tutor: agenda generation, replan, final response, revision."""
from __future__ import annotations

import json
from typing import Optional

from .utils import safe_json_loads, render_dialogue
from .prompts.meta_tutor_prompts import (
    META_TUTOR_AGENDA_SYSTEM,
    META_TUTOR_AGENDA_USER,
    META_TUTOR_REPLAN_SYSTEM,
    META_TUTOR_REPLAN_USER,
    META_TUTOR_FINAL_SYSTEM,
    META_TUTOR_FINAL_USER,
    META_TUTOR_REVISE_SYSTEM,
    META_TUTOR_REVISE_USER,
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


class MetaTutor:
    def __init__(self, client, temperature: float = 0.0, max_tokens: int = 1024):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

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
    ) -> str:
        messages = [
            {"role": "system", "content": META_TUTOR_FINAL_SYSTEM},
            {"role": "user", "content": META_TUTOR_FINAL_USER.format(
                problem=problem,
                dialogue=render_dialogue(dialogue) or "(empty)",
                worker_outputs=json.dumps(
                    {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in worker_outputs.items()},
                    ensure_ascii=False,
                ),
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return (raw or "").strip()

    def revise_final(
        self,
        problem: str,
        dialogue: list,
        worker_outputs: dict,
        draft: str,
        auditor_feedback: dict,
    ) -> str:
        fb = {k: v for k, v in auditor_feedback.items() if not k.startswith("_")}
        messages = [
            {"role": "system", "content": META_TUTOR_REVISE_SYSTEM},
            {"role": "user", "content": META_TUTOR_REVISE_USER.format(
                problem=problem,
                dialogue=render_dialogue(dialogue) or "(empty)",
                worker_outputs=json.dumps(
                    {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in worker_outputs.items()},
                    ensure_ascii=False,
                ),
                draft=draft,
                auditor_feedback=json.dumps(fb, ensure_ascii=False),
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return (raw or "").strip()
