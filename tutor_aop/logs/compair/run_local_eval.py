# -*- coding: utf-8 -*-
"""Run the dialogue-level holistic evaluation against a local open-source
LLM (served by vLLM with an OpenAI-compatible API) instead of GPT.

Reuses the same English judge prompt defined in ``make_request.py`` so
results are directly comparable to the GPT-judge runs. Iterates over every
``compair/dialogues/<model>.jsonl`` file, sends one chat-completions call
per problem, parses the returned JSON, and aggregates per-criterion mean
scores per model.

Outputs:
- ``compair/eval_results/<model>.jsonl``   -- per-row raw response + parsed scores
- ``compair/eval_summary.csv``             -- one row per model with mean scores
- table printed to stdout

Default judge endpoint matches ``config.yaml`` tutor_server: ``http://127.0.0.1:9000/v1``.

Usage
-----
    python tutor_aop/logs/compair/run_local_eval.py
    python tutor_aop/logs/compair/run_local_eval.py --model google/gemma-2-27b-it
    python tutor_aop/logs/compair/run_local_eval.py --num 8           # smoke test
    python tutor_aop/logs/compair/run_local_eval.py --concurrency 32
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Same-directory import (compair/) of the shared judge prompt.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from make_request import EVAL_SYSTEM_PROMPT, build_eval_prompt  # noqa: E402


CRITERIA = [
    "relevance",
    "usefulness",
    "clarity",
    "information_non_leakage",
    "socratic_engagement",
]
ALL_KEYS = CRITERIA + ["overall"]


# --------------------------------------------------------------------- IO


def _list_dialogue_files(in_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(in_dir)):
        if name.endswith(".jsonl") and os.path.isfile(os.path.join(in_dir, name)):
            out.append(os.path.join(in_dir, name))
    return out


def _read_dialogues(path: str, limit: Optional[int]) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ------------------------------------------------------------- JSON parse


def _try_parse_json(text: str) -> Optional[dict]:
    """Tolerant JSON extraction: strip code fences, find first {...} block."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # remove leading fence (with or without "json" lang)
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Fallback: find outermost JSON object by brace matching.
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_scores(parsed: Optional[dict]) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {k: None for k in ALL_KEYS}
    if not isinstance(parsed, dict):
        return scores
    ev = parsed.get("evaluation") or {}
    for k in CRITERIA:
        v = ev.get(k)
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
            scores[k] = float(v["score"])
    ov = parsed.get("overall") or {}
    if isinstance(ov.get("score"), (int, float)):
        scores["overall"] = float(ov["score"])
    return scores


# ------------------------------------------------------------- LLM client


def _make_client(base_url: str, api_key: str, timeout: int):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _judge_one(client, model: str, problem: str, gold: str, dialogue: list,
               max_tokens: int, retries: int) -> Tuple[Optional[str], Optional[str]]:
    """One chat call. Returns (raw_text, error_str). Exactly one is None."""
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_eval_prompt(problem, gold, dialogue)},
    ]
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("empty content")
            return content, None
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    return None, f"{type(last_err).__name__}: {last_err}"


# -------------------------------------------------------------- per file


def _process_file(
    src_path: str,
    out_dir: str,
    client,
    model: str,
    max_tokens: int,
    retries: int,
    concurrency: int,
    limit: Optional[int],
) -> Dict:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.jsonl")
    rows = _read_dialogues(src_path, limit)
    n = len(rows)
    if n == 0:
        return {"src": os.path.basename(src_path), "n": 0, "parse_fail": 0,
                "scores": {k: None for k in ALL_KEYS}}

    print(f"  [{stem}] dispatching {n} requests (concurrency={concurrency})...",
          file=sys.stderr, flush=True)

    results: List[Optional[Dict]] = [None] * n

    def _work(i: int) -> Tuple[int, Dict]:
        row = rows[i]
        problem = row.get("problem") or ""
        gold = row.get("gold_answer") or ""
        dlg = row.get("dialogue") or []
        raw, err = _judge_one(client, model, problem, gold, dlg, max_tokens, retries)
        parsed = _try_parse_json(raw) if raw is not None else None
        scores = _extract_scores(parsed)
        return i, {
            "custom_id": str(row.get("index")),
            "scores": scores,
            "parse_ok": parsed is not None,
            "error": err,
            "raw": raw,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_work, i) for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            i, item = fut.result()
            results[i] = item
            done += 1
            if done % 50 == 0 or done == n:
                print(f"  [{stem}] {done}/{n} done ({time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)

    # Persist per-row.
    with open(out_path, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Aggregate.
    parse_fail = sum(1 for r in results if not r["parse_ok"])
    agg: Dict[str, Optional[float]] = {}
    for k in ALL_KEYS:
        vals = [r["scores"][k] for r in results if r["scores"][k] is not None]
        agg[k] = round(statistics.mean(vals), 3) if vals else None
    agg_n = {k: sum(1 for r in results if r["scores"][k] is not None)
             for k in ALL_KEYS}

    return {
        "src": os.path.basename(src_path),
        "out": os.path.relpath(out_path, os.path.dirname(out_dir)),
        "n": n,
        "parse_fail": parse_fail,
        "scores": agg,
        "n_scored": agg_n,
    }


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "dialogues"),
                    help="Directory holding per-model dialogue JSONL files.")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "eval_results"),
                    help="Directory to write per-row eval JSONLs.")
    ap.add_argument("--summary-csv", default=os.path.join(HERE, "eval_summary.csv"),
                    help="CSV path for the aggregated per-model scores.")
    ap.add_argument("--base-url", default="http://127.0.0.1:9000/v1",
                    help="vLLM OpenAI-compatible endpoint.")
    ap.add_argument("--model", default="google/gemma-4-31B-it",
                    help="Judge model name as advertised by the vLLM server.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Judge response max tokens (JSON output is short).")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="Parallel in-flight requests against the judge.")
    ap.add_argument("--num", type=int, default=None,
                    help="Limit per-file rows (smoke testing).")
    ap.add_argument("--only", type=str, default=None,
                    help="Substring filter -- only process source files whose "
                         "name contains this substring.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    src_files = _list_dialogue_files(in_dir)
    if args.only:
        src_files = [p for p in src_files if args.only in os.path.basename(p)]
    if not src_files:
        print(f"[local_eval] no input .jsonl files in {in_dir}", file=sys.stderr)
        return

    client = _make_client(args.base_url, args.api_key, args.timeout)

    print(f"[local_eval] judge: {args.model} @ {args.base_url}", file=sys.stderr)
    print(f"[local_eval] in_dir={in_dir}", file=sys.stderr)
    print(f"[local_eval] out_dir={out_dir}", file=sys.stderr)
    print(f"[local_eval] sources: {len(src_files)}  concurrency={args.concurrency}",
          file=sys.stderr)
    print("", file=sys.stderr)

    summaries: List[Dict] = []
    for src in src_files:
        s = _process_file(
            src, out_dir, client, args.model,
            max_tokens=args.max_tokens, retries=args.retries,
            concurrency=args.concurrency, limit=args.num,
        )
        summaries.append(s)

    # Print + save summary table.
    cols = ["model"] + ALL_KEYS + ["n", "parse_fail"]
    print("", file=sys.stderr)
    print("=" * 100)
    header = (f"{'model':<55s}  " + "  ".join(f"{c:>10s}" for c in ALL_KEYS) +
              f"  {'n':>4s}  {'fail':>4s}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        stem = os.path.splitext(s["src"])[0]
        score_cells = []
        for k in ALL_KEYS:
            v = s["scores"][k]
            score_cells.append(f"{v:>10.3f}" if v is not None else f"{'-':>10s}")
        print(f"{stem[:55]:<55s}  " + "  ".join(score_cells) +
              f"  {s['n']:>4d}  {s['parse_fail']:>4d}")
    print("=" * 100)

    with open(args.summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in summaries:
            stem = os.path.splitext(s["src"])[0]
            row = [stem]
            for k in ALL_KEYS:
                v = s["scores"][k]
                row.append("" if v is None else f"{v:.3f}")
            row.extend([s["n"], s["parse_fail"]])
            w.writerow(row)
    print(f"\n[local_eval] summary -> {args.summary_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
