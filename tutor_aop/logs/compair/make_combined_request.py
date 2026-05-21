# -*- coding: utf-8 -*-
"""Build a SINGLE OpenAI Batch-API request file per judge model that bundles
every per-model dialogue file in ``compair/dialogues/`` together.

Reuses the system prompt and dialogue-rendering helpers from
``make_request.py`` (do not duplicate the rubric).

Output (under ``compair/eval_requests/``):
    gpt-4o-mini_all.jsonl   -- one batch covering all dialogue files
    gpt-5-mini_all.jsonl    -- one batch covering all dialogue files

custom_id format: ``{dialogue_stem}__{problem_index}`` so results can be
demuxed back to the source model.

Usage
-----
    python -m tutor_aop.logs.compair.make_combined_request
    python -m tutor_aop.logs.compair.make_combined_request --judges gpt-4o-mini
    python -m tutor_aop.logs.compair.make_combined_request \
        --in-dir <path> --out-dir <path>
"""
import argparse
import json
import os
import sys
from typing import Dict, List

from make_request import EVAL_SYSTEM_PROMPT, build_eval_prompt


HERE = os.path.dirname(os.path.abspath(__file__))


# Per-judge body parameters. Mirrors gpt4omini_request_exam.py and
# gpt5mini_request_exam.py exactly.
JUDGE_CONFIGS: Dict[str, dict] = {
    "gpt-4o-mini": {
        "max_completion_tokens": 1000,
        "extra": {"temperature": 0.0},
    },
    "gpt-5-mini": {
        "max_completion_tokens": 3000,
        "extra": {"reasoning_effort": "low"},
        # No temperature: gpt-5* only accepts the default.
    },
}


def _list_input_files(in_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".jsonl"):
            continue
        full = os.path.join(in_dir, name)
        if os.path.isfile(full):
            out.append(full)
    return out


def _iter_dialogue_rows(src_path: str):
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield row


def build_batch_for_judge(
    judge_model: str,
    src_files: List[str],
    out_dir: str,
) -> dict:
    cfg = JUDGE_CONFIGS[judge_model]
    out_path = os.path.join(out_dir, f"{judge_model}_all.jsonl")

    n_in = 0
    n_skipped = 0
    n_written = 0
    per_src: List[dict] = []

    with open(out_path, "w", encoding="utf-8") as f_out:
        for src in src_files:
            stem = os.path.splitext(os.path.basename(src))[0]
            src_in = src_skip = src_w = 0

            for row in _iter_dialogue_rows(src):
                src_in += 1
                n_in += 1

                problem = row.get("problem")
                gold = row.get("gold_answer")
                dlg = row.get("dialogue")
                if not (problem and gold and isinstance(dlg, list) and len(dlg) > 0):
                    src_skip += 1
                    n_skipped += 1
                    continue

                cid = f"{stem}__{row.get('index')}"

                messages = [
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": build_eval_prompt(problem, gold, dlg)},
                ]

                body = {
                    "model": judge_model,
                    "messages": messages,
                    "max_completion_tokens": cfg["max_completion_tokens"],
                }
                body.update(cfg["extra"])

                batch_item = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f_out.write(json.dumps(batch_item, ensure_ascii=False) + "\n")
                src_w += 1
                n_written += 1

            per_src.append({
                "src": os.path.basename(src),
                "in": src_in,
                "skipped": src_skip,
                "written": src_w,
            })

    return {
        "judge": judge_model,
        "out": out_path,
        "in": n_in,
        "skipped": n_skipped,
        "written": n_written,
        "per_src": per_src,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "dialogues"),
                    help="Directory holding per-model dialogue JSONL files.")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "eval_requests"),
                    help="Directory to write the combined batch files.")
    ap.add_argument("--judges", nargs="+",
                    default=["gpt-4o-mini", "gpt-5-mini"],
                    choices=list(JUDGE_CONFIGS.keys()),
                    help="Judge models to build batches for.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        print(f"[make_combined_request] in-dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    src_files = _list_input_files(in_dir)
    if not src_files:
        print(f"[make_combined_request] no .jsonl files in {in_dir}", file=sys.stderr)
        return

    print(f"[make_combined_request] in_dir={in_dir}", file=sys.stderr)
    print(f"[make_combined_request] out_dir={out_dir}", file=sys.stderr)
    print(f"[make_combined_request] sources ({len(src_files)}):", file=sys.stderr)
    for s in src_files:
        print(f"    - {os.path.basename(s)}", file=sys.stderr)
    print("", file=sys.stderr)

    for judge in args.judges:
        stats = build_batch_for_judge(judge, src_files, out_dir)
        print(f"[{stats['judge']}]  in={stats['in']}  skipped={stats['skipped']}  "
              f"written={stats['written']}  ->  "
              f"{os.path.relpath(stats['out'], os.path.dirname(out_dir))}",
              file=sys.stderr)
        for sub in stats["per_src"]:
            print(f"    {sub['src']:<40s}  in={sub['in']:>4}  "
                  f"skip={sub['skipped']:>3}  written={sub['written']:>4}",
                  file=sys.stderr)
        print("", file=sys.stderr)

    print(f"[make_combined_request] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
