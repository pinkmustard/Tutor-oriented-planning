"""Batched, stage-wise driver for the AOP tutoring pipeline.

Mirrors ``classroom.py``'s baseline pattern but unfolds each tutor turn into
~9 sub-stages instead of one ``tutor.respond``:

    PLAN -> DETECT -> [REPLAN]
         -> DIAGNOSIS -> TUTOR_MOVE -> RETRIEVAL
         -> FINAL -> AUDIT -> [REVISE]
         -> STUDENT_RESPOND

Within each sub-stage we gather only the conversations that are still active
*and* whose current agenda includes that worker (DIAGNOSIS / TUTOR_MOVE /
RETRIEVAL stages skip conversations whose agenda doesn't list them) and fan
the calls out via a ``ThreadPoolExecutor``. With a ``RoundRobinLLMClient``
sitting in front of two tutor replicas, the per-turn 7-call burst is split
across the two GPUs.

Conversations that hit ``<end_of_conversation>``, ``max_turns``, or any
exception drop into ``STUDENT_RESOLVE`` and stop participating in further
sub-stages, so each batch shrinks naturally turn-over-turn (same behavior as
the baseline classroom).
"""
from __future__ import annotations

import sys
import time
import traceback as _tb
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evaluator import grade
from .utils import contains_end_signal


class AopState(Enum):
    START = 0
    STUDENT_INITIAL = 1
    TUTOR_TURN = 2       # entry point for all per-turn tutor sub-stages
    STUDENT_TURN = 3     # student.respond after tutor finished its turn
    STUDENT_RESOLVE = 4
    END = 5


@dataclass
class AopConv:
    row: dict
    state: AopState = AopState.START
    initial_solution: Optional[str] = None
    initial_grade: Optional[dict] = None
    tutoring_needed: bool = False
    dialogue: List[dict] = field(default_factory=list)
    turn_logs: List[dict] = field(default_factory=list)
    turn_idx: int = -1

    # Per-turn scratch (cleared at start of every TUTOR_TURN entry).
    curr_agenda: Optional[dict] = None         # raw plan_agenda output
    curr_active_agenda: Optional[dict] = None  # post-replan (or = curr_agenda)
    curr_detector: Optional[dict] = None
    curr_diagnosis: dict = field(default_factory=dict)
    curr_worker_outputs: dict = field(default_factory=dict)
    curr_draft: Optional[str] = None
    curr_audit: Optional[dict] = None
    curr_final: Optional[str] = None

    ended_by: Optional[str] = None
    post_tutoring_solution: Optional[str] = None
    post_tutoring_grade: Optional[dict] = None
    fatal_error: Optional[str] = None
    fatal_traceback: Optional[str] = None
    elapsed_sec: float = 0.0
    _t0: float = 0.0


ErrTuple = Tuple[str, str, str]  # (exc_name, message, traceback_str)


def _run_parallel(
    fn: Callable[[Any], Any],
    items: List[Any],
    max_workers: int,
) -> List[Tuple[Optional[Any], Optional[ErrTuple]]]:
    """Same helper shape as ``classroom._run_parallel``; duplicated here to
    avoid an import cycle and keep the AOP driver self-contained."""
    if not items:
        return []

    results: List[Optional[Tuple[Optional[Any], Optional[ErrTuple]]]] = [None] * len(items)

    def _wrap(i: int, item: Any):
        try:
            return i, fn(item), None
        except Exception as e:  # noqa: BLE001
            return i, None, (type(e).__name__, str(e), _tb.format_exc())

    workers = max(1, min(max_workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_wrap, i, it) for i, it in enumerate(items)]
        for f in futs:
            i, out, err = f.result()
            results[i] = (out, err)

    return [r if r is not None else (None, None) for r in results]


def _log(msg: str) -> None:
    print(f"[aop_classroom] {msg}", file=sys.stderr, flush=True)


def _strip_underscores(d):
    """Drop keys starting with '_' (e.g. ``_raw``) before logging."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _has_worker(agenda: Optional[dict], worker_name: str) -> bool:
    if not agenda:
        return False
    for item in agenda.get("agenda", []) or []:
        if item.get("worker") == worker_name:
            return True
    return False


def _fresh_turn_log(turn_idx: int) -> dict:
    return {
        "turn_idx": turn_idx,
        "initial_agenda": None,
        "detector_output": None,
        "replan_agenda": None,
        "executed_agenda": None,
        "worker_outputs": None,
        "draft_response": None,
        "auditor_output": None,
        "revised_response": None,
        "final_tutor_response": None,
        "student_response": None,
        "errors": [],
    }


def _record_error(c: AopConv, turn_log: dict, err: ErrTuple, turn_idx: int) -> None:
    """Mark a conv as errored mid-turn and route it to STUDENT_RESOLVE so we
    still attempt a post-tutoring solve."""
    turn_log["errors"].append(f"{err[0]}: {err[1]}")
    turn_log["errors"].append(err[2])
    if c.ended_by is None:
        c.ended_by = f"error_in_turn_{turn_idx}"
    c.state = AopState.STUDENT_RESOLVE


def _reset_turn_scratch(c: AopConv) -> None:
    c.curr_agenda = None
    c.curr_active_agenda = None
    c.curr_detector = None
    c.curr_diagnosis = {}
    c.curr_worker_outputs = {}
    c.curr_draft = None
    c.curr_audit = None
    c.curr_final = None


def run_aop_batch(
    rows: List[dict],
    pipe: dict,
    cfg: dict,
    fixed_initials: Optional[Dict[int, str]] = None,
) -> List[AopConv]:
    """Stage-wise AOP runner.

    Stages:
      1. ``student.initial_solve`` (or fixed-initials lookup) for all rows.
      2. Per-turn loop: PLAN -> DETECT -> [REPLAN] -> DIAGNOSIS ->
         TUTOR_MOVE -> RETRIEVAL -> FINAL -> AUDIT -> [REVISE] ->
         STUDENT_RESPOND, each as one ``_run_parallel`` batch over the
         active subset.
      3. ``student.independent_resolve`` for everyone that needed tutoring.
    """
    exp = cfg["experiment"]
    max_turns: int = exp["max_turns"]
    max_replan: int = exp["max_replan"]
    max_revision: int = exp["max_revision"]
    concurrency: int = int(exp.get("concurrency", 32))

    meta_tutor = pipe["meta_tutor"]
    detector = pipe["detector"]
    auditor = pipe["auditor"]
    workers = pipe["workers"]
    student = pipe["student"]

    convs: List[AopConv] = [AopConv(row=r) for r in rows]
    now = time.time()
    for c in convs:
        c._t0 = now
        c.state = AopState.STUDENT_INITIAL

    # --- Stage 1: initial_solve --------------------------------------------
    active_initial = [c for c in convs if c.state == AopState.STUDENT_INITIAL]

    if fixed_initials is not None:
        _log(f"initial_solve (fixed): {len(active_initial)} problems")
        t0 = time.time()
        for c in active_initial:
            idx = c.row.get("index")
            if idx is None or idx not in fixed_initials:
                c.fatal_error = (
                    f"fixed_initial_solutions has no entry for idx={idx!r}"
                )
                c.fatal_traceback = ""
                c.state = AopState.END
                c.ended_by = "fatal_error_initial"
                continue
            out = fixed_initials[idx]
            c.initial_solution = out
            c.initial_grade = grade(out, c.row.get("answer", ""))
            if c.initial_grade.get("correct"):
                c.state = AopState.END
                c.ended_by = "skip_correct_initial"
            else:
                c.tutoring_needed = True
                c.dialogue = [{"role": "student", "content": out}]
                c.state = AopState.TUTOR_TURN
        _log(f"initial_solve (fixed) done in {time.time() - t0:.2f}s; "
             f"tutoring_needed={sum(c.tutoring_needed for c in convs)}")
    else:
        _log(f"initial_solve: {len(active_initial)} problems (concurrency={concurrency})")
        t0 = time.time()
        results = _run_parallel(
            lambda c: student.initial_solve(c.row["problem"]),
            active_initial,
            concurrency,
        )
        for c, (out, err) in zip(active_initial, results):
            if err is not None:
                c.fatal_error = f"{err[0]}: {err[1]}"
                c.fatal_traceback = err[2]
                c.state = AopState.END
                c.ended_by = "fatal_error_initial"
                continue
            c.initial_solution = out
            c.initial_grade = grade(out, c.row.get("answer", ""))
            if c.initial_grade.get("correct"):
                c.state = AopState.END
                c.ended_by = "skip_correct_initial"
            else:
                c.tutoring_needed = True
                c.dialogue = [{"role": "student", "content": out}]
                c.state = AopState.TUTOR_TURN
        _log(f"initial_solve done in {time.time() - t0:.1f}s; "
             f"tutoring_needed={sum(c.tutoring_needed for c in convs)}")

    # --- Stage 2: per-turn AOP pipeline ------------------------------------
    for turn_idx in range(max_turns):
        turn_active = [c for c in convs if c.state == AopState.TUTOR_TURN]
        if not turn_active:
            break

        # Open a new turn log + reset per-turn scratch.
        for c in turn_active:
            _reset_turn_scratch(c)
            c.turn_logs.append(_fresh_turn_log(turn_idx))
            c.turn_idx = turn_idx

        # ----- PLAN -----
        _log(f"turn {turn_idx}: PLAN batch={len(turn_active)}")
        t0 = time.time()
        results = _run_parallel(
            lambda c: meta_tutor.plan_agenda(
                problem=c.row["problem"],
                dialogue=c.dialogue,
                turn_idx=turn_idx,
                max_turns=max_turns,
            ),
            turn_active,
            concurrency,
        )
        for c, (out, err) in zip(turn_active, results):
            tl = c.turn_logs[-1]
            if err is not None:
                _record_error(c, tl, err, turn_idx)
                continue
            c.curr_agenda = out
            tl["initial_agenda"] = _strip_underscores(out)
        _log(f"turn {turn_idx}: PLAN done in {time.time() - t0:.1f}s")

        # ----- DETECT -----
        detect_active = [
            c for c in turn_active
            if c.state == AopState.TUTOR_TURN and c.curr_agenda is not None
        ]
        if detect_active:
            _log(f"turn {turn_idx}: DETECT batch={len(detect_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: detector.detect(
                    c.row["problem"], c.curr_agenda, turn_idx, max_turns,
                ),
                detect_active,
                concurrency,
            )
            for c, (out, err) in zip(detect_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                c.curr_detector = out
                tl["detector_output"] = _strip_underscores(out)
            _log(f"turn {turn_idx}: DETECT done in {time.time() - t0:.1f}s")

        # ----- REPLAN (subset: only those flagged by detector) -----
        if max_replan > 0:
            replan_active = [
                c for c in turn_active
                if c.state == AopState.TUTOR_TURN
                and c.curr_detector is not None
                and detector.needs_replan(c.curr_detector)
            ]
        else:
            replan_active = []
        if replan_active:
            _log(f"turn {turn_idx}: REPLAN batch={len(replan_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: meta_tutor.replan(
                    c.row["problem"], c.curr_agenda, c.curr_detector, c.dialogue,
                ),
                replan_active,
                concurrency,
            )
            for c, (out, err) in zip(replan_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                c.curr_active_agenda = out
                tl["replan_agenda"] = _strip_underscores(out)
            _log(f"turn {turn_idx}: REPLAN done in {time.time() - t0:.1f}s")

        # Promote active agenda; default to initial agenda.
        for c in turn_active:
            if c.state != AopState.TUTOR_TURN:
                continue
            if c.curr_active_agenda is None:
                c.curr_active_agenda = c.curr_agenda
            tl = c.turn_logs[-1]
            tl["executed_agenda"] = _strip_underscores(c.curr_active_agenda)

        # ----- WORKERS: DIAGNOSIS -----
        diag_active = [
            c for c in turn_active
            if c.state == AopState.TUTOR_TURN
            and _has_worker(c.curr_active_agenda, "diagnosis")
        ]
        if diag_active and "diagnosis" in workers:
            _log(f"turn {turn_idx}: DIAGNOSIS batch={len(diag_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: workers["diagnosis"].run(c.row["problem"], c.dialogue),
                diag_active,
                concurrency,
            )
            for c, (out, err) in zip(diag_active, results):
                if err is not None:
                    c.curr_worker_outputs["diagnosis"] = {
                        "error": f"{err[0]}: {err[1]}"
                    }
                    continue
                c.curr_diagnosis = out
                c.curr_worker_outputs.setdefault("diagnosis", out)
            _log(f"turn {turn_idx}: DIAGNOSIS done in {time.time() - t0:.1f}s")

        # ----- WORKERS: TUTOR_MOVE -----
        move_active = [
            c for c in turn_active
            if c.state == AopState.TUTOR_TURN
            and _has_worker(c.curr_active_agenda, "tutor_move")
        ]
        if move_active and "tutor_move" in workers:
            _log(f"turn {turn_idx}: TUTOR_MOVE batch={len(move_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: workers["tutor_move"].run(
                    c.row["problem"], c.curr_diagnosis, c.dialogue,
                ),
                move_active,
                concurrency,
            )
            for c, (out, err) in zip(move_active, results):
                if err is not None:
                    c.curr_worker_outputs["tutor_move"] = {
                        "error": f"{err[0]}: {err[1]}"
                    }
                    continue
                c.curr_worker_outputs.setdefault("tutor_move", out)
            _log(f"turn {turn_idx}: TUTOR_MOVE done in {time.time() - t0:.1f}s")

        # ----- WORKERS: RETRIEVAL -----
        retr_active = [
            c for c in turn_active
            if c.state == AopState.TUTOR_TURN
            and _has_worker(c.curr_active_agenda, "retrieval")
        ]
        if retr_active and workers.get("retrieval") is not None:
            _log(f"turn {turn_idx}: RETRIEVAL batch={len(retr_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: workers["retrieval"].run(
                    c.row["problem"], c.curr_diagnosis, c.dialogue,
                ),
                retr_active,
                concurrency,
            )
            for c, (out, err) in zip(retr_active, results):
                if err is not None:
                    c.curr_worker_outputs["retrieval"] = {
                        "error": f"{err[0]}: {err[1]}"
                    }
                    continue
                c.curr_worker_outputs.setdefault("retrieval", out)
            _log(f"turn {turn_idx}: RETRIEVAL done in {time.time() - t0:.1f}s")

        for c in turn_active:
            if c.state != AopState.TUTOR_TURN:
                continue
            tl = c.turn_logs[-1]
            tl["worker_outputs"] = {
                k: _strip_underscores(v) for k, v in c.curr_worker_outputs.items()
            }

        # ----- GENERATE_FINAL -----
        final_active = [c for c in turn_active if c.state == AopState.TUTOR_TURN]
        if final_active:
            _log(f"turn {turn_idx}: FINAL batch={len(final_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: meta_tutor.generate_final(
                    problem=c.row["problem"],
                    dialogue=c.dialogue,
                    worker_outputs=c.curr_worker_outputs,
                ),
                final_active,
                concurrency,
            )
            for c, (out, err) in zip(final_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                c.curr_draft = out
                tl["draft_response"] = out
            _log(f"turn {turn_idx}: FINAL done in {time.time() - t0:.1f}s")

        # ----- AUDIT -----
        audit_active = [
            c for c in turn_active
            if c.state == AopState.TUTOR_TURN and c.curr_draft is not None
        ]
        if audit_active:
            _log(f"turn {turn_idx}: AUDIT batch={len(audit_active)}")
            t0 = time.time()

            def _audit(c: AopConv):
                tutor_move = ""
                tm = c.curr_worker_outputs.get("tutor_move")
                if isinstance(tm, dict):
                    tutor_move = tm.get("selected_move", "") or ""
                return auditor.audit(
                    c.row["problem"], c.curr_draft, tutor_move=tutor_move,
                )

            results = _run_parallel(_audit, audit_active, concurrency)
            for c, (out, err) in zip(audit_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                c.curr_audit = out
                tl["auditor_output"] = _strip_underscores(out)
            _log(f"turn {turn_idx}: AUDIT done in {time.time() - t0:.1f}s")

        # ----- REVISE (subset: only auditor-flagged drafts) -----
        if max_revision > 0:
            revise_active = [
                c for c in turn_active
                if c.state == AopState.TUTOR_TURN
                and c.curr_audit is not None
                and auditor.needs_revision(c.curr_audit)
            ]
        else:
            revise_active = []
        if revise_active:
            _log(f"turn {turn_idx}: REVISE batch={len(revise_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: meta_tutor.revise_final(
                    problem=c.row["problem"],
                    dialogue=c.dialogue,
                    worker_outputs=c.curr_worker_outputs,
                    draft=c.curr_draft,
                    auditor_feedback=c.curr_audit,
                ),
                revise_active,
                concurrency,
            )
            for c, (out, err) in zip(revise_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                tl["revised_response"] = out
                c.curr_final = out
            _log(f"turn {turn_idx}: REVISE done in {time.time() - t0:.1f}s")

        # Pick final response, append to dialogue, advance state.
        for c in turn_active:
            if c.state != AopState.TUTOR_TURN:
                continue
            tl = c.turn_logs[-1]
            final = c.curr_final if c.curr_final is not None else c.curr_draft
            if final is None:
                _record_error(
                    c, tl,
                    ("RuntimeError", "no final response produced", ""),
                    turn_idx,
                )
                continue
            tl["final_tutor_response"] = final
            c.dialogue.append({"role": "tutor", "content": final})
            if contains_end_signal(final):
                c.ended_by = "end_token"
                c.state = AopState.STUDENT_RESOLVE
            else:
                c.state = AopState.STUDENT_TURN

        # ----- STUDENT_RESPOND -----
        student_active = [c for c in convs if c.state == AopState.STUDENT_TURN]
        if student_active:
            _log(f"turn {turn_idx}: STUDENT batch={len(student_active)}")
            t0 = time.time()
            results = _run_parallel(
                lambda c: student.respond(
                    problem=c.row["problem"], dialogue=c.dialogue,
                ),
                student_active,
                concurrency,
            )
            for c, (out, err) in zip(student_active, results):
                tl = c.turn_logs[-1]
                if err is not None:
                    _record_error(c, tl, err, turn_idx)
                    continue
                tl["student_response"] = out
                c.dialogue.append({"role": "student", "content": out})
                if turn_idx == max_turns - 1:
                    c.ended_by = "max_turns"
                    c.state = AopState.STUDENT_RESOLVE
                else:
                    c.state = AopState.TUTOR_TURN
            _log(f"turn {turn_idx}: STUDENT done in {time.time() - t0:.1f}s")

    # Safety: anyone still mid-turn after the budget gets promoted.
    for c in convs:
        if c.state in (AopState.TUTOR_TURN, AopState.STUDENT_TURN):
            if c.ended_by is None:
                c.ended_by = "max_turns"
            c.state = AopState.STUDENT_RESOLVE

    # --- Stage 3: independent_resolve --------------------------------------
    resolve_active = [c for c in convs if c.state == AopState.STUDENT_RESOLVE]
    _log(f"independent_resolve: {len(resolve_active)} conversations")
    t0 = time.time()
    results = _run_parallel(
        lambda c: student.independent_resolve(c.row["problem"], c.dialogue),
        resolve_active,
        concurrency,
    )
    for c, (out, err) in zip(resolve_active, results):
        if err is not None:
            c.post_tutoring_solution = None
            c.post_tutoring_grade = {
                "correct": False,
                "reason": f"{err[0]}: {err[1]}",
            }
        else:
            c.post_tutoring_solution = out
            c.post_tutoring_grade = grade(out, c.row.get("answer", ""))
        c.state = AopState.END
    _log(f"independent_resolve done in {time.time() - t0:.1f}s")

    end = time.time()
    for c in convs:
        c.elapsed_sec = round(end - c._t0, 3)

    return convs


def aop_conv_to_log_row(
    c: AopConv,
    tutor_model: str,
    student_model: str,
) -> dict:
    """Serialize an ``AopConv`` to the JSONL schema the analysis tooling
    consumes. Same shape as ``classroom.conv_to_log_row`` plus the AOP-only
    per-turn keys (initial_agenda, detector_output, ..., final_tutor_response)."""
    row: dict = {
        "index": c.row.get("index"),
        "problem": c.row.get("problem"),
        "gold_answer": c.row.get("answer"),
        "level": c.row.get("level"),
        "subject": c.row.get("subject"),
        "initial_solution": c.initial_solution,
        "initial_grade": c.initial_grade,
        "tutoring_needed": c.tutoring_needed,
        "turns": c.turn_logs,
        "ended_by": c.ended_by,
        "post_tutoring_solution": c.post_tutoring_solution,
        "post_tutoring_grade": c.post_tutoring_grade,
        "elapsed_sec": c.elapsed_sec,
        "tutor_model": tutor_model,
        "student_model": student_model,
        "system": "aop",
    }
    if c.tutoring_needed:
        row["dialogue"] = c.dialogue
    if c.fatal_error is not None:
        row["fatal_error"] = c.fatal_error
        row["traceback"] = c.fatal_traceback
    return row
