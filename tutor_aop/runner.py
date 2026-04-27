"""Orchestrator for the detector-guided, AOP-inspired tutoring pipeline.

Per problem, runs:
  1. Student initial solve (skip if already correct).
  2. Multi-turn tutoring:
       Meta-Tutor.agenda -> Detector -> (Replan x1) -> Workers ->
       Meta-Tutor.final  -> Auditor  -> (Revision x1) -> Student.respond
  3. Student independent re-solve using the dialogue transcript.
  4. Logging (one JSONL row per episode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Optional

import yaml
from tqdm import tqdm

from .llm_client import build_clients_from_config
from .meta_tutor import MetaTutor
from .detector import PlanDetector
from .auditor import PedagogicalAuditor
from .student import StudentAgent
from .workers import DiagnosisWorker, TutorMoveWorker, RetrievalWorker
from .evaluator import grade
from .utils import contains_end_signal, extract_boxed


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
    return cfg


def load_dataset(cfg: dict, mock: bool) -> list:
    name = cfg["experiment"]["dataset"]
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
            },
            {
                "problem": "Find the area of a right triangle with legs 3 and 4.",
                "solution": "Area = (1/2)(3)(4) = 6.",
                "answer": "6",
                "level": "Level 1",
                "subject": "Geometry",
            },
        ][:max(1, num)]

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


def build_pipeline(cfg: dict):
    tutor_client, student_client, vllm_manager = build_clients_from_config(cfg)

    meta_tutor = MetaTutor(
        client=tutor_client,
        temperature=cfg["experiment"]["temperature"],
        max_tokens=cfg["experiment"]["max_tokens"],
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
        max_tokens=exp_cfg["max_tokens"],
        resolve_max_tokens=exp_cfg.get("student_resolve_max_tokens"),
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


def _execute_agenda(agenda: dict, workers: dict, problem: str,
                    dialogue: list) -> dict:
    """Run workers in order; diagnosis feeds tutor_move & retrieval when present."""
    outputs = {}
    diag = {}
    for item in agenda.get("agenda", []):
        w_name = item.get("worker")
        subtask = item.get("task", "")
        worker = workers.get(w_name)
        if worker is None:
            continue
        try:
            if w_name == "diagnosis":
                out = worker.run(problem, dialogue, subtask=subtask)
                diag = out
            elif w_name == "tutor_move":
                out = worker.run(problem, diag, dialogue, subtask=subtask)
            elif w_name == "retrieval":
                out = worker.run(problem, diag, dialogue, subtask=subtask)
            else:
                continue
            outputs.setdefault(w_name, out)
        except Exception as e:
            outputs[w_name] = {"error": f"{type(e).__name__}: {e}"}
    return outputs


def run_episode(problem_row: dict, pipe: dict, cfg: dict) -> dict:
    exp = cfg["experiment"]
    max_turns = exp["max_turns"]
    max_replan = exp["max_replan"]
    max_revision = exp["max_revision"]

    problem = problem_row["problem"]
    gold = problem_row["answer"]

    meta_tutor = pipe["meta_tutor"]
    detector = pipe["detector"]
    auditor = pipe["auditor"]
    workers = pipe["workers"]
    student = pipe["student"]

    episode = {
        "index": problem_row.get("index"),
        "problem": problem,
        "gold_answer": gold,
        "level": problem_row.get("level"),
        "subject": problem_row.get("subject"),
        "initial_solution": None,
        "initial_grade": None,
        "tutoring_needed": False,
        "turns": [],
        "ended_by": None,
        "post_tutoring_solution": None,
        "post_tutoring_grade": None,
    }

    initial = student.initial_solve(problem)
    episode["initial_solution"] = initial
    episode["initial_grade"] = grade(initial, gold)

    if episode["initial_grade"]["correct"]:
        episode["ended_by"] = "skip_correct_initial"
        return episode

    episode["tutoring_needed"] = True

    dialogue = [{"role": "student", "content": initial}]

    for turn_idx in range(max_turns):
        turn_log = {
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

        try:
            agenda = meta_tutor.plan_agenda(
                problem=problem,
                dialogue=dialogue,
                turn_idx=turn_idx,
                max_turns=max_turns,
            )
            turn_log["initial_agenda"] = {k: v for k, v in agenda.items() if not k.startswith("_")}

            det = detector.detect(problem, agenda, turn_idx, max_turns)
            turn_log["detector_output"] = {k: v for k, v in det.items() if not k.startswith("_")}

            active_agenda = agenda
            if detector.needs_replan(det) and max_replan > 0:
                new_agenda = meta_tutor.replan(problem, agenda, det, dialogue)
                turn_log["replan_agenda"] = {k: v for k, v in new_agenda.items() if not k.startswith("_")}
                active_agenda = new_agenda

            turn_log["executed_agenda"] = {k: v for k, v in active_agenda.items() if not k.startswith("_")}

            worker_outs = _execute_agenda(
                active_agenda, workers, problem, dialogue,
            )
            turn_log["worker_outputs"] = {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in worker_outs.items()
            }

            draft = meta_tutor.generate_final(
                problem=problem,
                dialogue=dialogue,
                worker_outputs=worker_outs,
            )
            turn_log["draft_response"] = draft

            tutor_move_str = ""
            if "tutor_move" in worker_outs:
                tutor_move_str = worker_outs["tutor_move"].get("selected_move", "")

            audit = auditor.audit(problem, draft, tutor_move=tutor_move_str)
            turn_log["auditor_output"] = {k: v for k, v in audit.items() if not k.startswith("_")}

            final_response = draft
            if auditor.needs_revision(audit) and max_revision > 0:
                revised = meta_tutor.revise_final(
                    problem=problem,
                    dialogue=dialogue,
                    worker_outputs=worker_outs,
                    draft=draft,
                    auditor_feedback=audit,
                )
                turn_log["revised_response"] = revised
                final_response = revised

            turn_log["final_tutor_response"] = final_response
            dialogue.append({"role": "tutor", "content": final_response})

            if contains_end_signal(final_response):
                episode["turns"].append(turn_log)
                episode["ended_by"] = "end_token"
                break

            student_resp = student.respond(
                problem=problem,
                dialogue=dialogue,
            )
            turn_log["student_response"] = student_resp
            dialogue.append({"role": "student", "content": student_resp})

            episode["turns"].append(turn_log)

        except Exception as e:
            turn_log["errors"].append(f"{type(e).__name__}: {e}")
            turn_log["errors"].append(traceback.format_exc())
            episode["turns"].append(turn_log)
            episode["ended_by"] = f"error_in_turn_{turn_idx}"
            break

    if episode["ended_by"] is None:
        episode["ended_by"] = "max_turns"

    try:
        post = student.independent_resolve(problem, dialogue)
        episode["post_tutoring_solution"] = post
        episode["post_tutoring_grade"] = grade(post, gold)
    except Exception as e:
        episode["post_tutoring_solution"] = None
        episode["post_tutoring_grade"] = {"correct": False, "reason": f"{type(e).__name__}: {e}"}

    episode["dialogue"] = dialogue
    return episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--mock", action="store_true", help="Run with deterministic mock backend (no vLLM).")
    ap.add_argument("--num", type=int, default=None, help="Override experiment.num_problems.")
    ap.add_argument("--start", type=int, default=None, help="Override experiment.start_index.")
    ap.add_argument("--out", type=str, default=None, help="Override log file path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        cfg.setdefault("mock", {})["enabled"] = True
    if args.num is not None:
        cfg["experiment"]["num_problems"] = args.num
    if args.start is not None:
        cfg["experiment"]["start_index"] = args.start

    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = args.out or os.path.join(log_dir, cfg["logging"]["log_filename"])

    data = load_dataset(cfg, mock=cfg.get("mock", {}).get("enabled", False))
    pipe = build_pipeline(cfg)

    print(f"[runner] problems={len(data)}  log={log_path}  mock={cfg.get('mock',{}).get('enabled', False)}",
          file=sys.stderr)

    n_correct_initial = 0
    n_correct_post = 0
    n_tutored = 0

    try:
        with open(log_path, "a", encoding="utf-8") as fout:
            for row in tqdm(data, file=sys.stderr):
                t0 = time.time()
                try:
                    episode = run_episode(row, pipe, cfg)
                except Exception as e:
                    episode = {
                        "index": row.get("index"),
                        "problem": row.get("problem"),
                        "gold_answer": row.get("answer"),
                        "fatal_error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                episode["elapsed_sec"] = round(time.time() - t0, 3)
                fout.write(json.dumps(episode, ensure_ascii=False) + "\n")
                fout.flush()

                if episode.get("initial_grade", {}).get("correct"):
                    n_correct_initial += 1
                if episode.get("tutoring_needed"):
                    n_tutored += 1
                    if episode.get("post_tutoring_grade", {}).get("correct"):
                        n_correct_post += 1

        total = len(data)
        print(f"[runner] DONE  total={total}  initial_correct={n_correct_initial}  "
              f"tutored={n_tutored}  post_tutoring_correct={n_correct_post}",
              file=sys.stderr)
    finally:
        mgr = pipe.get("vllm_manager")
        if mgr is not None:
            mgr.shutdown()


if __name__ == "__main__":
    main()
