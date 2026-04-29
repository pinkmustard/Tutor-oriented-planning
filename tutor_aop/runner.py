"""AOP runner: stage-wise batched detector-guided pipeline across the whole
problem set, with the same shape as ``baseline_runner`` (one batch call,
results streamed to JSONL at the end).

Per turn the pipeline unfolds into ~9 sub-stages -- see ``aop_classroom`` for
the full state diagram:

    Student.initial_solve (N problems)              -- batch on student server
    Loop turn 1..max_turns:
        Meta-Tutor.plan_agenda                      -- batch on tutor servers (RR)
        Detector.detect                             -- batch on tutor servers
        [Meta-Tutor.replan]   (subset)              -- batch on tutor servers
        DiagnosisWorker / TutorMoveWorker /
            RetrievalWorker  (per-agenda subsets)   -- batches on tutor servers
        Meta-Tutor.generate_final                   -- batch on tutor servers
        Auditor.audit                               -- batch on tutor servers
        [Meta-Tutor.revise_final] (subset)          -- batch on tutor servers
        Student.respond  (active subset)            -- batch on student server
    Student.independent_resolve (all tutored)       -- batch on student server

vLLM continuous-batching packs each stage's concurrent requests server-side.
With multiple tutor replicas behind a ``RoundRobinLLMClient``, the per-turn
~7-call burst is split across GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

from .llm_client import build_clients_from_config
from .meta_tutor import MetaTutor
from .detector import PlanDetector
from .auditor import PedagogicalAuditor
from .student import StudentAgent
from .workers import DiagnosisWorker, TutorMoveWorker, RetrievalWorker
from .classroom import load_fixed_initials
from .aop_classroom import run_aop_batch, aop_conv_to_log_row


HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pool_path = cfg.get("retrieval", {}).get("pool_path", "")
    if pool_path and not os.path.isabs(pool_path):
        cfg["retrieval"]["pool_path"] = os.path.join(HERE, pool_path)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    if log_dir and not os.path.isabs(log_dir):
        cfg["logging"]["log_dir"] = os.path.join(HERE, log_dir)
    fixed = cfg.get("experiment", {}).get("fixed_initial_solutions")
    if fixed and not os.path.isabs(fixed):
        cfg["experiment"]["fixed_initial_solutions"] = os.path.join(HERE, fixed)
    return cfg


def load_dataset(cfg: dict, mock: bool) -> list:
    """Load problems from the configured HF dataset/split. See baseline_runner
    for field-map notes (Big-Math has no solution/level/subject)."""
    name = cfg["experiment"]["dataset"]
    split = cfg["experiment"].get("dataset_split", "test")
    num = cfg["experiment"]["num_problems"]
    start = cfg["experiment"].get("start_index", 0)

    if mock:
        return [
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
        ][:max(1, num)]

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


def build_pipeline(cfg: dict):
    tutor_client, student_client, vllm_manager = build_clients_from_config(cfg)

    meta_tutor = MetaTutor(
        client=tutor_client,
        temperature=cfg["experiment"]["temperature"],
        max_tokens=cfg["experiment"]["max_tokens"],
        tutor_turn_max_tokens=cfg["experiment"]["tutor_turn_max_tokens"],
    )
    detector = PlanDetector(client=tutor_client)
    auditor = PedagogicalAuditor(client=tutor_client)
    diagnosis_w = DiagnosisWorker(client=tutor_client)
    move_w = TutorMoveWorker(client=tutor_client)

    retrieval_w = None
    if cfg.get("retrieval", {}).get("enabled", True):
        retrieval_w = RetrievalWorker(
            client=tutor_client,
            pool_path=cfg["retrieval"]["pool_path"],
            top_k=cfg["retrieval"].get("top_k", 3),
        )

    exp_cfg = cfg["experiment"]
    student = StudentAgent(
        client=student_client,
        temperature=exp_cfg.get("student_temperature", 0.7),
        initial_max_tokens=exp_cfg["student_initial_max_tokens"],
        respond_max_tokens=exp_cfg["student_respond_max_tokens"],
        resolve_max_tokens=exp_cfg["student_resolve_max_tokens"],
    )

    return {
        "tutor_client": tutor_client,
        "student_client": student_client,
        "meta_tutor": meta_tutor,
        "detector": detector,
        "auditor": auditor,
        "workers": {
            "diagnosis": diagnosis_w,
            "tutor_move": move_w,
            "retrieval": retrieval_w,
        },
        "student": student,
        "vllm_manager": vllm_manager,
    }


def _slugify_model(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--mock", action="store_true",
                    help="Run with deterministic mock backend (no vLLM).")
    ap.add_argument("--num", type=int, default=None,
                    help="Override experiment.num_problems.")
    ap.add_argument("--start", type=int, default=None,
                    help="Override experiment.start_index.")
    ap.add_argument("--tutor-model", type=str, default=None,
                    help="Override tutor_server.model.")
    ap.add_argument("--student-model", type=str, default=None,
                    help="Override student_server.model.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override experiment.dataset (HF dataset name).")
    ap.add_argument("--dataset-split", type=str, default=None,
                    help="Override experiment.dataset_split (default: test).")
    ap.add_argument("--fixed-initials", type=str, default=None,
                    help="Override experiment.fixed_initial_solutions "
                         "(path relative to tutor_aop/, or absolute, or 'none' to disable).")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override experiment.concurrency (thread-pool width per stage).")
    ap.add_argument("--out", type=str, default=None,
                    help="Override log file path. Default: logs/aop_<tutor>.jsonl")
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
    default_name = f"aop_{tutor_slug}"
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
            f"[aop] fixed_initial_solutions: {fixed_path} "
            f"({len(fixed_initials)} entries) -- skipping live initial_solve",
            file=sys.stderr,
        )

    pipe = build_pipeline(cfg)

    exp = cfg["experiment"]
    print(
        f"[aop] tutor={cfg['tutor_server']['model']}  "
        f"student={cfg['student_server']['model']}",
        file=sys.stderr,
    )
    print(
        f"[aop] problems={len(data)}  log={log_path}  "
        f"mock={cfg.get('mock', {}).get('enabled', False)}  "
        f"concurrency={exp.get('concurrency', 32)}",
        file=sys.stderr,
    )

    try:
        t_all = time.time()
        convs = run_aop_batch(
            rows=data,
            pipe=pipe,
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
                row_out = aop_conv_to_log_row(c, tutor_model, student_model)
                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                if (c.initial_grade or {}).get("correct"):
                    n_correct_initial += 1
                if c.tutoring_needed:
                    n_tutored += 1
                    if (c.post_tutoring_grade or {}).get("correct"):
                        n_correct_post += 1

        total = len(convs)
        print(
            f"[aop] DONE  total={total}  initial_correct={n_correct_initial}  "
            f"tutored={n_tutored}  post_tutoring_correct={n_correct_post}  "
            f"wall={elapsed_all:.1f}s",
            file=sys.stderr,
        )
    finally:
        mgr = pipe.get("vllm_manager")
        if mgr is not None:
            mgr.shutdown()


if __name__ == "__main__":
    main()
