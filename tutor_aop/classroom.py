"""Batched, state-machine driven tutor-student classroom.

Modeled after PedagogicalRL's `src/classroom.py` (sample_conversations loop),
trimmed to the baseline use case: ATTEMPTED-only conversations, no judge.

Per-conversation state machine:
    START -> STUDENT_INITIAL -> (END if initial correct)
                             -> TUTOR_TURN <-> STUDENT_TURN (alternating)
                             -> STUDENT_RESOLVE -> END

The whole batch is driven stage-by-stage. At each stage we gather the
conversations whose `state` matches, fan the LLM calls out over a
ThreadPoolExecutor (vLLM handles request batching server-side), collect the
results, then advance each conversation's state. Conversations that terminate
early (correct initial answer, <end_of_conversation> token, max_turns, error)
simply fall out of the active set for subsequent stages, so a batch of 500
shrinks naturally turn-over-turn.
"""
from __future__ import annotations

import sys
import time
import traceback as _tb
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from .evaluator import grade
from .utils import contains_end_signal


class ConvState(Enum):
    START = 0
    STUDENT_INITIAL = 1
    TUTOR_TURN = 2
    STUDENT_TURN = 3
    STUDENT_RESOLVE = 4
    END = 5


@dataclass
class BaselineConv:
    row: dict
    state: ConvState = ConvState.START
    initial_solution: Optional[str] = None
    initial_grade: Optional[dict] = None
    tutoring_needed: bool = False
    dialogue: List[dict] = field(default_factory=list)
    turn_logs: List[dict] = field(default_factory=list)
    turn_idx: int = -1
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
    """Map ``fn`` over ``items`` with a thread pool, preserving order.

    Returns a list of ``(result, error)`` tuples aligned with ``items``.
    Exactly one side is non-None per entry. Errors are captured as
    ``(exc_name, message, traceback_str)``.
    """
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
    print(f"[classroom] {msg}", file=sys.stderr, flush=True)


def run_baseline_batch(
    rows: List[dict],
    tutor,
    student,
    cfg: dict,
) -> List[BaselineConv]:
    """Run the whole baseline experiment as stage-wise batched calls.

    Stages:
      1. ``student.initial_solve`` for all rows.
      2. Multi-turn loop: ``tutor.respond`` batch -> ``student.respond`` batch,
         while any conversation is still in an active turn state. Each
         conversation independently transitions to STUDENT_RESOLVE when it hits
         `<end_of_conversation>`, max_turns, or an error.
      3. ``student.independent_resolve`` for every conversation that needed
         tutoring (or that hit an error during tutoring -- we still want a
         post-tutoring grade attempt when possible).

    vLLM-side parallelism is exactly the same as in PedagogicalRL: we issue N
    concurrent HTTP calls during one stage, the server's continuous batcher
    packs them into GPU batches. Tutor and student sit on separate vLLM
    servers and (on single-GPU setups) are swapped via the VLLMManager
    sleep/wake mechanism -- the swap happens once per stage, not per request.
    """
    exp = cfg["experiment"]
    max_turns: int = exp["max_turns"]
    concurrency: int = int(exp.get("concurrency", 32))

    convs: List[BaselineConv] = [BaselineConv(row=r) for r in rows]
    now = time.time()
    for c in convs:
        c._t0 = now
        c.state = ConvState.STUDENT_INITIAL

    # --- Stage 1: initial_solve -------------------------------------------
    active_initial = [c for c in convs if c.state == ConvState.STUDENT_INITIAL]
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
            c.state = ConvState.END
            c.ended_by = "fatal_error_initial"
            continue
        c.initial_solution = out
        c.initial_grade = grade(out, c.row.get("answer", ""))
        if c.initial_grade.get("correct"):
            c.state = ConvState.END
            c.ended_by = "skip_correct_initial"
        else:
            c.tutoring_needed = True
            c.dialogue = [{"role": "student", "content": out}]
            c.state = ConvState.TUTOR_TURN
    _log(f"initial_solve done in {time.time() - t0:.1f}s; "
         f"tutoring_needed={sum(c.tutoring_needed for c in convs)}")

    # --- Stage 2: multi-turn loop -----------------------------------------
    for turn_idx in range(max_turns):
        tutor_active = [c for c in convs if c.state == ConvState.TUTOR_TURN]
        if not tutor_active:
            break

        # Tutor turn
        _log(f"turn {turn_idx}: tutor batch size={len(tutor_active)}")
        t0 = time.time()
        tutor_results = _run_parallel(
            lambda c: tutor.respond(c.row["problem"], c.dialogue),
            tutor_active,
            concurrency,
        )
        for c, (out, err) in zip(tutor_active, tutor_results):
            turn_log = {
                "turn_idx": turn_idx,
                "tutor_response": None,
                "student_response": None,
                "errors": [],
            }
            c.turn_logs.append(turn_log)
            c.turn_idx = turn_idx

            if err is not None:
                turn_log["errors"].append(f"{err[0]}: {err[1]}")
                turn_log["errors"].append(err[2])
                c.ended_by = f"error_in_turn_{turn_idx}"
                c.state = ConvState.STUDENT_RESOLVE
                continue

            turn_log["tutor_response"] = out
            c.dialogue.append({"role": "tutor", "content": out})
            if contains_end_signal(out):
                c.ended_by = "end_token"
                c.state = ConvState.STUDENT_RESOLVE
            else:
                c.state = ConvState.STUDENT_TURN
        _log(f"turn {turn_idx}: tutor done in {time.time() - t0:.1f}s")

        # Student turn
        student_active = [c for c in convs if c.state == ConvState.STUDENT_TURN]
        if not student_active:
            continue

        _log(f"turn {turn_idx}: student batch size={len(student_active)}")
        t0 = time.time()
        student_results = _run_parallel(
            lambda c: student.respond(
                problem=c.row["problem"],
                dialogue=c.dialogue,
            ),
            student_active,
            concurrency,
        )
        for c, (out, err) in zip(student_active, student_results):
            turn_log = c.turn_logs[-1]
            if err is not None:
                turn_log["errors"].append(f"{err[0]}: {err[1]}")
                turn_log["errors"].append(err[2])
                c.ended_by = f"error_in_turn_{turn_idx}"
                c.state = ConvState.STUDENT_RESOLVE
                continue
            turn_log["student_response"] = out
            c.dialogue.append({"role": "student", "content": out})
            if turn_idx == max_turns - 1:
                # Student answered on the final allowed turn; we've exhausted
                # the budget -- the tutor will not speak again.
                c.ended_by = "max_turns"
                c.state = ConvState.STUDENT_RESOLVE
            else:
                c.state = ConvState.TUTOR_TURN
        _log(f"turn {turn_idx}: student done in {time.time() - t0:.1f}s")

    # Safety: anyone still lingering in TUTOR_TURN/STUDENT_TURN after the
    # budget is exhausted gets promoted to STUDENT_RESOLVE.
    for c in convs:
        if c.state in (ConvState.TUTOR_TURN, ConvState.STUDENT_TURN):
            if c.ended_by is None:
                c.ended_by = "max_turns"
            c.state = ConvState.STUDENT_RESOLVE

    # --- Stage 3: independent_resolve -------------------------------------
    resolve_active = [c for c in convs if c.state == ConvState.STUDENT_RESOLVE]
    _log(f"independent_resolve: {len(resolve_active)} conversations")
    t0 = time.time()
    resolve_results = _run_parallel(
        lambda c: student.independent_resolve(c.row["problem"], c.dialogue),
        resolve_active,
        concurrency,
    )
    for c, (out, err) in zip(resolve_active, resolve_results):
        if err is not None:
            c.post_tutoring_solution = None
            c.post_tutoring_grade = {
                "correct": False,
                "reason": f"{err[0]}: {err[1]}",
            }
        else:
            c.post_tutoring_solution = out
            c.post_tutoring_grade = grade(out, c.row.get("answer", ""))
        c.state = ConvState.END
    _log(f"independent_resolve done in {time.time() - t0:.1f}s")

    end = time.time()
    for c in convs:
        c.elapsed_sec = round(end - c._t0, 3)

    return convs


def conv_to_log_row(
    c: BaselineConv,
    tutor_model: str,
    student_model: str,
) -> dict:
    """Serialize a BaselineConv into the JSONL schema the analysis tooling
    already consumes (same keys as the pre-refactor baseline_runner)."""
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
        "system": "baseline",
    }
    if c.tutoring_needed:
        row["dialogue"] = c.dialogue
    if c.fatal_error is not None:
        row["fatal_error"] = c.fatal_error
        row["traceback"] = c.fatal_traceback
    return row
