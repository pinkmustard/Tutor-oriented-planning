"""Baseline runner: pure tutor-student multi-turn dialogue with batched,
stage-wise execution across the whole problem set.

Matches the PedagogicalRL classroom flow (ATTEMPTED-only, judge/reward
stripped) so one vLLM batch drives many conversations concurrently instead of
one problem at a time:

    Student.initial_solve (N problems)          -- one batch on student server
    Loop turn 1..max_turns:
        Tutor.respond    (active tutor_turn subset)   -- one batch on tutor
        Student.respond  (active student_turn subset) -- one batch on student
    Student.independent_resolve (all tutored)  -- one batch on student server

Conversations that hit `<end_of_conversation>`, max_turns, or correct-initial
drop out of the active set naturally, so the turn batches shrink over time.

Episode schema on disk is identical to the pre-refactor output; analysis
tooling does not need to change.

Swap tutor models via `--tutor-model`. Because a swap loads fresh weights,
run one invocation per tutor model:

    for m in Qwen/Qwen2.5-7B-Instruct CogBase-USTC/SocraticLM eth-nlped/TutorRL-7B; do
        python -m tutor_aop.baseline_runner --tutor-model "$m"
    done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

from .llm_client import build_clients_from_config
from .baseline_tutor import BaselineTutor
from .student import StudentAgent
from .classroom import run_baseline_batch, conv_to_log_row


HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if log_dir and not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(HERE, log_dir)
    return cfg


def load_dataset(cfg: dict, mock: bool) -> list:
    name = cfg["experiment"]["dataset"]
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
            },
            {
                "problem": "Find the area of a right triangle with legs 3 and 4.",
                "solution": "Area = (1/2)(3)(4) = 6.",
                "answer": "6",
                "level": "Level 1",
                "subject": "Geometry",
            },
        ]
        return base[: max(1, num)]

    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(name, split="test")
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
                    help="Override tutor_server.model (e.g. CogBase-USTC/SocraticLM).")
    ap.add_argument("--student-model", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override experiment.concurrency (thread-pool width).")
    ap.add_argument("--out", type=str, default=None,
                    help="Override log file path. Default: logs/baseline_<tutor>.jsonl")
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
    if args.tutor_model is not None:
        cfg["tutor_server"]["model"] = args.tutor_model
    if args.student_model is not None:
        cfg["student_server"]["model"] = args.student_model
    if args.concurrency is not None:
        cfg["experiment"]["concurrency"] = args.concurrency

    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    tutor_slug = _slugify_model(cfg["tutor_server"]["model"])
    default_name = f"baseline_{tutor_slug}"
    if args.tag:
        default_name += f"_{args.tag}"
    default_name += ".jsonl"
    log_path = args.out or os.path.join(log_dir, default_name)

    data = load_dataset(cfg, mock=cfg.get("mock", {}).get("enabled", False))

    tutor_client, student_client, vllm_manager = build_clients_from_config(cfg)

    exp = cfg["experiment"]
    tutor = BaselineTutor(
        client=tutor_client,
        temperature=exp["temperature"],
        max_tokens=exp["max_tokens"],
    )
    student = StudentAgent(
        client=student_client,
        temperature=exp.get("student_temperature", 0.7),
        max_tokens=exp["max_tokens"],
        resolve_max_tokens=exp.get("student_resolve_max_tokens"),
    )

    print(
        f"[baseline] tutor={cfg['tutor_server']['model']}  "
        f"student={cfg['student_server']['model']}",
        file=sys.stderr,
    )
    print(
        f"[baseline] problems={len(data)}  log={log_path}  "
        f"mock={cfg.get('mock', {}).get('enabled', False)}  "
        f"concurrency={exp.get('concurrency', 32)}",
        file=sys.stderr,
    )

    try:
        t_all = time.time()
        convs = run_baseline_batch(
            rows=data,
            tutor=tutor,
            student=student,
            cfg=cfg,
        )
        elapsed_all = time.time() - t_all

        tutor_model = cfg["tutor_server"]["model"]
        student_model = cfg["student_server"]["model"]

        n_correct_initial = 0
        n_tutored = 0
        n_correct_post = 0
        with open(log_path, "a", encoding="utf-8") as fout:
            for c in convs:
                row_out = conv_to_log_row(c, tutor_model, student_model)
                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                if (c.initial_grade or {}).get("correct"):
                    n_correct_initial += 1
                if c.tutoring_needed:
                    n_tutored += 1
                    if (c.post_tutoring_grade or {}).get("correct"):
                        n_correct_post += 1

        total = len(convs)
        print(
            f"[baseline] DONE  total={total}  initial_correct={n_correct_initial}  "
            f"tutored={n_tutored}  post_tutoring_correct={n_correct_post}  "
            f"wall={elapsed_all:.1f}s",
            file=sys.stderr,
        )
    finally:
        if vllm_manager is not None:
            vllm_manager.shutdown()


if __name__ == "__main__":
    main()
