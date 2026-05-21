"""Extract tutoring dialogues from comparison logs into a slim, shared schema.

Reads every ``*.jsonl`` episode log sitting next to this script (the
``compair/`` folder) and, for each source file, writes a per-model ``.jsonl``
containing only:

    index, problem, gold_answer, level, subject, dialogue

Episodes whose initial student attempt was already correct are skipped --
no tutoring happened for them, so there's nothing to compare. ``dialogue``
is preserved as-is from the source log (begins with the student's initial
solution, followed by alternating tutor / student turns).

Output folder: ``compair/dialogues/`` by default. One ``.jsonl`` per input
log, named ``<source_stem>.jsonl``.

Usage
-----
    python -m tutor_aop.logs.compair.extract_dialogues
    python -m tutor_aop.logs.compair.extract_dialogues --out-format json
    python -m tutor_aop.logs.compair.extract_dialogues --in-dir <path> --out-dir <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List


HERE = os.path.dirname(os.path.abspath(__file__))


def _is_tutored(row: dict) -> bool:
    """Episode counts as tutored only if the initial attempt was wrong AND a
    dialogue actually exists. ``tutoring_needed`` alone may be True for rows
    that errored out before any turn, which produce no dialogue -- guarding
    on the dialogue list itself is safer."""
    if not row.get("tutoring_needed"):
        return False
    dlg = row.get("dialogue")
    return isinstance(dlg, list) and len(dlg) > 0


def _slim_row(row: dict) -> dict:
    """Project the source row onto the comparison schema."""
    return {
        "index": row.get("index"),
        "problem": row.get("problem"),
        "gold_answer": row.get("gold_answer"),
        "level": row.get("level"),
        "subject": row.get("subject"),
        "dialogue": row.get("dialogue"),
    }


def _list_input_files(in_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".jsonl"):
            continue
        if name.startswith("extract"):
            continue
        full = os.path.join(in_dir, name)
        if os.path.isfile(full):
            files.append(full)
    return files


def _write_jsonl(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def extract_one(src_path: str, out_dir: str, out_format: str) -> dict:
    """Process one source log, writing one output file. Returns stats."""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    ext = ".json" if out_format == "json" else ".jsonl"
    out_path = os.path.join(out_dir, f"{stem}{ext}")

    n_total = 0
    n_skipped_correct = 0
    n_skipped_no_dialogue = 0
    kept: List[dict] = []

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if (row.get("initial_grade") or {}).get("correct"):
                n_skipped_correct += 1
                continue
            if not _is_tutored(row):
                n_skipped_no_dialogue += 1
                continue
            kept.append(_slim_row(row))

    if out_format == "json":
        _write_json(kept, out_path)
    else:
        _write_jsonl(kept, out_path)

    return {
        "src": os.path.basename(src_path),
        "out": os.path.relpath(out_path, os.path.dirname(out_dir)),
        "total": n_total,
        "skipped_correct_initial": n_skipped_correct,
        "skipped_no_dialogue": n_skipped_no_dialogue,
        "kept": len(kept),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=HERE,
                    help="Directory holding source episode JSONL files. "
                         "Default: this script's folder.")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "dialogues"),
                    help="Directory to write extracted per-model files. "
                         "Default: <in-dir>/dialogues/")
    ap.add_argument("--out-format", choices=["jsonl", "json"], default="jsonl",
                    help="Output file format (default: jsonl).")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    src_files = _list_input_files(in_dir)
    if not src_files:
        print(f"[extract] no .jsonl files found in {in_dir}", file=sys.stderr)
        return

    print(f"[extract] in_dir={in_dir}", file=sys.stderr)
    print(f"[extract] out_dir={out_dir}", file=sys.stderr)
    print(f"[extract] sources: {len(src_files)}", file=sys.stderr)
    print("", file=sys.stderr)

    rows = []
    for src in src_files:
        stats = extract_one(src, out_dir, args.out_format)
        rows.append(stats)
        print(
            f"  {stats['src']:<60s}  total={stats['total']:>3}  "
            f"skip_correct={stats['skipped_correct_initial']:>3}  "
            f"skip_no_dlg={stats['skipped_no_dialogue']:>3}  "
            f"kept={stats['kept']:>3}  ->  {stats['out']}",
            file=sys.stderr,
        )

    print("", file=sys.stderr)
    print(f"[extract] DONE  files written: {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
