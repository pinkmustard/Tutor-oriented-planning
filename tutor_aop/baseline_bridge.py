"""Bridge baseline: tutor reply per turn = task1 -> task2 -> task3 chain.

Multi-turn flow is identical to ``baseline_runner`` (perspective-rotated
dialogue, ATTEMPTED-only, fixed-initial student solutions, batched stages),
but each tutor turn is a 3-step bridge produced against the tutor server:

    Task1 (infer student error type) ->
    Task2 (pick remediation strategy + intention) ->
    Task3 (generate the actual short tutor utterance, conditioned on
           Task1 + Task2 outputs and the conversation so far).

Only the Task3 output goes back into the dialogue (so the student model
never sees the diagnostic JSON). Task1 / Task2 outputs are saved per turn
in the JSONL log for later analysis.

Two-GPU layout matches ``config.yaml``: tutor on the external 96GB GPU
(vLLM @ 9000), student on the local A6000 (vLLM @ 8001), no sleep/wake
swapping (``vllm_manager.enabled: false``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import yaml

from .classroom import (
    BaselineConv,
    ConvState,
    _run_parallel,
    load_fixed_initials,
)
from .evaluator import grade
from .llm_client import build_clients_from_config
from .prompts.baseline_bridge_prompt import (
    TASK1_INFER_ERROR_SYSTEM,
    TASK1_INFER_ERROR_USER,
    TASK2_STRATEGY_SYSTEM,
    TASK2_STRATEGY_USER,
    TASK3_GENERATE_RESPONSE_SYSTEM,
    TASK3_GENERATE_RESPONSE_USER,
)
from .student import StudentAgent
from .utils import contains_end_signal, render_dialogue


HERE = os.path.dirname(os.path.abspath(__file__))


class BaselineBridgeTutor:
    """3-step tutor: error inference -> strategy/intention -> response.

    Exposes the three stages as separate methods (``task1``, ``task2``,
    ``task3``) so the batched runner can fan them out as distinct
    sub-stages instead of running them sequentially per worker thread.
    Task1 and Task2 are independent (both consume only ``problem`` +
    ``c_h``), so the runner fires them as one combined 2N-item wave;
    Task3 depends on both and runs in a second N-item wave.
    """

    def __init__(
        self,
        client,
        temperature: float = 0.7,
        task_diag_max_tokens: int = 1024,
        task_response_max_tokens: int = 320,
    ):
        self.client = client
        self.temperature = temperature
        self.task_diag_max_tokens = task_diag_max_tokens
        self.task_response_max_tokens = task_response_max_tokens

    def task1(self, problem: str, dialogue: List[Dict[str, str]]) -> str:
        c_h = render_dialogue(dialogue)
        messages = [
            {"role": "system", "content": TASK1_INFER_ERROR_SYSTEM},
            {
                "role": "user",
                "content": TASK1_INFER_ERROR_USER.format(problem=problem, c_h=c_h),
            },
        ]
        return (self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.task_diag_max_tokens,
        ) or "").strip()

    def task2(self, problem: str, dialogue: List[Dict[str, str]]) -> str:
        c_h = render_dialogue(dialogue)
        messages = [
            {"role": "system", "content": TASK2_STRATEGY_SYSTEM},
            {
                "role": "user",
                "content": TASK2_STRATEGY_USER.format(problem=problem, c_h=c_h),
            },
        ]
        return (self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.task_diag_max_tokens,
        ) or "").strip()

    def task3(
        self,
        problem: str,
        dialogue: List[Dict[str, str]],
        task1_response: str,
        task2_response: str,
    ) -> str:
        c_h = render_dialogue(dialogue)
        messages = [
            {"role": "system", "content": TASK3_GENERATE_RESPONSE_SYSTEM},
            {
                "role": "user",
                "content": TASK3_GENERATE_RESPONSE_USER.format(
                    task1_response=task1_response,
                    task2_response=task2_response,
                    problem=problem,
                    c_h=c_h,
                ),
            },
        ]
        return (self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.task_response_max_tokens,
        ) or "").strip()

    def respond(
        self,
        problem: str,
        dialogue: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """Single-shot convenience path (sequential task1 -> task2 -> task3).

        The batched runner does NOT use this; it calls task1/task2/task3
        directly so it can stage-wise batch them. Kept for ad-hoc use
        (smoke tests, REPL).
        """
        t1 = self.task1(problem, dialogue)
        t2 = self.task2(problem, dialogue)
        final = self.task3(problem, dialogue, t1, t2)
        return {"final": final, "task1": t1, "task2": t2}


def _log(msg: str) -> None:
    print(f"[bridge] {msg}", file=sys.stderr, flush=True)


def run_bridge_batch(
    rows: List[dict],
    tutor: BaselineBridgeTutor,
    student: StudentAgent,
    cfg: dict,
    fixed_initials: Optional[Dict[int, str]] = None,
) -> List[BaselineConv]:
    """Stage-wise batched bridge runner.

    Same state machine as ``classroom.run_baseline_batch`` but each tutor
    turn fans out three sequential LLM calls per conv (task1, task2,
    task3) inside the tutor stage's ThreadPoolExecutor. Only task3 enters
    the dialogue; task1 / task2 outputs are stashed in the per-turn log.
    """
    exp = cfg["experiment"]
    max_turns: int = exp["max_turns"]
    concurrency: int = int(exp.get("concurrency", 32))

    convs: List[BaselineConv] = [BaselineConv(row=r) for r in rows]
    now = time.time()
    for c in convs:
        c._t0 = now
        c.state = ConvState.STUDENT_INITIAL

    # --- Stage 1: initial_solve --------------------------------------------
    active_initial = [c for c in convs if c.state == ConvState.STUDENT_INITIAL]

    if fixed_initials is not None:
        _log(f"initial_solve (fixed): looking up {len(active_initial)} problems")
        t0 = time.time()
        for c in active_initial:
            idx = c.row.get("index")
            if idx is None or idx not in fixed_initials:
                c.fatal_error = (
                    f"fixed_initial_solutions has no entry for idx={idx!r}"
                )
                c.fatal_traceback = ""
                c.state = ConvState.END
                c.ended_by = "fatal_error_initial"
                continue
            out = fixed_initials[idx]
            c.initial_solution = out
            c.initial_grade = grade(out, c.row.get("answer", ""))
            if c.initial_grade.get("correct"):
                c.state = ConvState.END
                c.ended_by = "skip_correct_initial"
            else:
                c.tutoring_needed = True
                c.dialogue = [{"role": "student", "content": out}]
                c.state = ConvState.TUTOR_TURN
        _log(
            f"initial_solve (fixed) done in {time.time() - t0:.2f}s; "
            f"tutoring_needed={sum(c.tutoring_needed for c in convs)}"
        )
    else:
        _log(
            f"initial_solve: {len(active_initial)} problems "
            f"(concurrency={concurrency})"
        )
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
        _log(
            f"initial_solve done in {time.time() - t0:.1f}s; "
            f"tutoring_needed={sum(c.tutoring_needed for c in convs)}"
        )

    # --- Stage 2: multi-turn loop ------------------------------------------
    for turn_idx in range(max_turns):
        tutor_active = [c for c in convs if c.state == ConvState.TUTOR_TURN]
        if not tutor_active:
            break

        # Allocate the per-turn log up-front for every active conv so both
        # diag substages and the task3 substage write into the same row.
        for c in tutor_active:
            c.turn_logs.append({
                "turn_idx": turn_idx,
                "tutor_response": None,
                "task1_output": None,
                "task2_output": None,
                "student_response": None,
                "errors": [],
            })
            c.turn_idx = turn_idx

        # ---- Substage A: task1 + task2 (independent) as one 2N batch ----
        # task1 / task2 only consume ``problem`` + ``c_h`` so they are
        # independent of each other. Combining them into a single thread
        # pool of 2N items lets the vLLM continuous batcher mix both
        # prompt families in one wave -- shared {problem}{c_h} prefix
        # also benefits prefix-cache hits across the two task types.
        diag_jobs = []
        for c in tutor_active:
            diag_jobs.append((c, "task1"))
            diag_jobs.append((c, "task2"))

        def _run_diag(job):
            cv, kind = job
            if kind == "task1":
                return tutor.task1(cv.row["problem"], cv.dialogue)
            return tutor.task2(cv.row["problem"], cv.dialogue)

        _log(
            f"turn {turn_idx}: diag batch size={len(diag_jobs)} "
            f"(task1+task2 fused, {len(tutor_active)} convs)"
        )
        t0 = time.time()
        diag_results = _run_parallel(_run_diag, diag_jobs, concurrency)
        _log(f"turn {turn_idx}: diag done in {time.time() - t0:.1f}s")

        # Pivot results back per-conv.
        per_conv_outputs: Dict[int, Dict[str, object]] = {}
        for (cv, kind), (out, err) in zip(diag_jobs, diag_results):
            slot = per_conv_outputs.setdefault(id(cv), {"task1": None, "task2": None,
                                                        "task1_err": None, "task2_err": None})
            if err is not None:
                slot[f"{kind}_err"] = err
            else:
                slot[kind] = out

        # Write diag outputs / errors into the turn_log; convs whose diag
        # stage failed are dropped to STUDENT_RESOLVE and skip task3.
        task3_inputs = []
        for c in tutor_active:
            slot = per_conv_outputs[id(c)]
            turn_log = c.turn_logs[-1]
            t1, t2 = slot["task1"], slot["task2"]
            t1_err, t2_err = slot["task1_err"], slot["task2_err"]

            if t1_err is not None or t2_err is not None:
                if t1_err is not None:
                    turn_log["errors"].append(f"task1: {t1_err[0]}: {t1_err[1]}")
                    turn_log["errors"].append(t1_err[2])
                if t2_err is not None:
                    turn_log["errors"].append(f"task2: {t2_err[0]}: {t2_err[1]}")
                    turn_log["errors"].append(t2_err[2])
                # Save whatever partial output we got for analysis.
                turn_log["task1_output"] = t1
                turn_log["task2_output"] = t2
                c.ended_by = f"error_in_turn_{turn_idx}"
                c.state = ConvState.STUDENT_RESOLVE
                continue

            turn_log["task1_output"] = t1
            turn_log["task2_output"] = t2
            task3_inputs.append((c, t1, t2))

        # ---- Substage B: task3 (depends on task1+task2) as one N batch ---
        if task3_inputs:
            _log(
                f"turn {turn_idx}: task3 batch size={len(task3_inputs)}"
            )
            t0 = time.time()
            task3_results = _run_parallel(
                lambda triple: tutor.task3(
                    triple[0].row["problem"],
                    triple[0].dialogue,
                    triple[1],
                    triple[2],
                ),
                task3_inputs,
                concurrency,
            )
            _log(f"turn {turn_idx}: task3 done in {time.time() - t0:.1f}s")

            for (c, _, _), (out, err) in zip(task3_inputs, task3_results):
                turn_log = c.turn_logs[-1]
                if err is not None:
                    turn_log["errors"].append(f"task3: {err[0]}: {err[1]}")
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
                c.ended_by = "max_turns"
                c.state = ConvState.STUDENT_RESOLVE
            else:
                c.state = ConvState.TUTOR_TURN
        _log(f"turn {turn_idx}: student done in {time.time() - t0:.1f}s")

    for c in convs:
        if c.state in (ConvState.TUTOR_TURN, ConvState.STUDENT_TURN):
            if c.ended_by is None:
                c.ended_by = "max_turns"
            c.state = ConvState.STUDENT_RESOLVE

    # --- Stage 3: independent_resolve --------------------------------------
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


def conv_to_log_row_bridge(
    c: BaselineConv,
    tutor_model: str,
    student_model: str,
) -> dict:
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
        "system": "baseline_bridge",
    }
    if c.tutoring_needed:
        row["dialogue"] = c.dialogue
    if c.fatal_error is not None:
        row["fatal_error"] = c.fatal_error
        row["traceback"] = c.fatal_traceback
    return row


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if log_dir and not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(HERE, log_dir)
    fixed = cfg.get("experiment", {}).get("fixed_initial_solutions")
    if fixed and not os.path.isabs(fixed):
        cfg["experiment"]["fixed_initial_solutions"] = os.path.join(HERE, fixed)
    return cfg


def load_dataset(cfg: dict, mock: bool) -> list:
    name = cfg["experiment"]["dataset"]
    split = cfg["experiment"].get("dataset_split", "test")
    num = cfg["experiment"]["num_problems"]
    start = cfg["experiment"].get("start_index", 0)

    if mock:
        base = [
            {
                "problem": "Solve for $x$: $2x + 3 = 9$.",
                "solution": "Subtract 3: $2x = 6$. Divide: $x = 3$.",
                "answer": "3",
                "level": "Level 1",
                "subject": "Algebra",
                "index": 0,
            },
            {
                "problem": "Find the area of a right triangle with legs 3 and 4.",
                "solution": "Area = (1/2)(3)(4) = 6.",
                "answer": "6",
                "level": "Level 1",
                "subject": "Geometry",
                "index": 1,
            },
        ]
        return base[: max(1, num)]

    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(name, split=split)
    end = min(start + num, len(ds))
    out = []
    for i in range(start, end):
        row = ds[i]
        out.append({
            "problem": row.get("problem"),
            "solution": row.get("solution"),
            "answer": row.get("answer"),
            "level": row.get("level"),
            "subject": row.get("subject"),
            "source": row.get("source"),
            "domain": row.get("domain"),
            "index": i,
        })
    return out


def _slugify_model(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--mock", action="store_true",
                    help="Run with deterministic mock backend (no vLLM).")
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--tutor-model", type=str, default=None,
                    help="Override tutor_server.model.")
    ap.add_argument("--student-model", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override experiment.dataset (HF dataset name).")
    ap.add_argument("--dataset-split", type=str, default=None,
                    help="Override experiment.dataset_split (default: test).")
    ap.add_argument("--fixed-initials", type=str, default=None,
                    help="Override experiment.fixed_initial_solutions "
                         "(path relative to tutor_aop/, or absolute, or 'none' to disable).")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override experiment.concurrency (thread-pool width).")
    ap.add_argument("--out", type=str, default=None,
                    help="Override log file path. Default: logs/baseline_bridge_<tutor>.jsonl")
    ap.add_argument("--tag", type=str, default=None,
                    help="Extra tag appended to default log filename.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        cfg.setdefault("mock", {})["enabled"] = True
    if args.num is not None:
        cfg["experiment"]["num_problems"] = args.num
    if args.start is not None:
        cfg["experiment"]["start_index"] = args.start
    if args.dataset is not None:
        cfg["experiment"]["dataset"] = args.dataset
    if args.dataset_split is not None:
        cfg["experiment"]["dataset_split"] = args.dataset_split
    if args.fixed_initials is not None:
        if args.fixed_initials.lower() == "none":
            cfg["experiment"]["fixed_initial_solutions"] = None
        else:
            p = args.fixed_initials
            if not os.path.isabs(p):
                p = os.path.join(HERE, p)
            cfg["experiment"]["fixed_initial_solutions"] = p
    if args.tutor_model is not None:
        cfg["tutor_server"]["model"] = args.tutor_model
    if args.student_model is not None:
        cfg["student_server"]["model"] = args.student_model
    if args.concurrency is not None:
        cfg["experiment"]["concurrency"] = args.concurrency

    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    tutor_slug = _slugify_model(cfg["tutor_server"]["model"])
    default_name = f"baseline_bridge_{tutor_slug}"
    if args.tag:
        default_name += f"_{args.tag}"
    default_name += ".jsonl"
    log_path = args.out or os.path.join(log_dir, default_name)

    data = load_dataset(cfg, mock=cfg.get("mock", {}).get("enabled", False))

    fixed_initials = None
    fixed_path = cfg["experiment"].get("fixed_initial_solutions")
    if fixed_path:
        fixed_initials = load_fixed_initials(fixed_path)
        print(
            f"[bridge] fixed_initial_solutions: {fixed_path} "
            f"({len(fixed_initials)} entries) -- skipping live initial_solve",
            file=sys.stderr,
        )

    tutor_client, student_client, vllm_manager = build_clients_from_config(cfg)

    exp = cfg["experiment"]
    tutor = BaselineBridgeTutor(
        client=tutor_client,
        temperature=exp["temperature"],
        task_diag_max_tokens=exp.get("max_tokens", 1024),
        task_response_max_tokens=exp["tutor_turn_max_tokens"],
    )
    student = StudentAgent(
        client=student_client,
        temperature=exp.get("student_temperature", 0.7),
        initial_max_tokens=exp["student_initial_max_tokens"],
        respond_max_tokens=exp["student_respond_max_tokens"],
        resolve_max_tokens=exp["student_resolve_max_tokens"],
    )

    print(
        f"[bridge] tutor={cfg['tutor_server']['model']}  "
        f"student={cfg['student_server']['model']}",
        file=sys.stderr,
    )
    print(
        f"[bridge] problems={len(data)}  log={log_path}  "
        f"mock={cfg.get('mock', {}).get('enabled', False)}  "
        f"concurrency={exp.get('concurrency', 32)}",
        file=sys.stderr,
    )

    try:
        t_all = time.time()
        convs = run_bridge_batch(
            rows=data,
            tutor=tutor,
            student=student,
            cfg=cfg,
            fixed_initials=fixed_initials,
        )
        elapsed_all = time.time() - t_all

        tutor_model = cfg["tutor_server"]["model"]
        student_model = cfg["student_server"]["model"]

        n_correct_initial = 0
        n_tutored = 0
        n_correct_post = 0
        with open(log_path, "a", encoding="utf-8") as fout:
            for c in convs:
                row_out = conv_to_log_row_bridge(c, tutor_model, student_model)
                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                if (c.initial_grade or {}).get("correct"):
                    n_correct_initial += 1
                if c.tutoring_needed:
                    n_tutored += 1
                    if (c.post_tutoring_grade or {}).get("correct"):
                        n_correct_post += 1

        total = len(convs)
        print(
            f"[bridge] DONE  total={total}  initial_correct={n_correct_initial}  "
            f"tutored={n_tutored}  post_tutoring_correct={n_correct_post}  "
            f"wall={elapsed_all:.1f}s",
            file=sys.stderr,
        )
    finally:
        if vllm_manager is not None:
            vllm_manager.shutdown()


if __name__ == "__main__":
    main()
