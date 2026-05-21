# -*- coding: utf-8 -*-
"""Build OpenAI Batch-API requests that ask a GPT judge to evaluate an
entire tutor-student dialogue holistically.

Reads the per-model dialogue files produced by ``extract_dialogues.py``
(``compair/dialogues/<model>.jsonl``) and, for each source file, writes a
sibling batch JSONL (``compair/eval_requests/<model>.jsonl``) containing
ONE request per problem.

Each request gives the judge:
- the math problem
- the gold answer
- the full tutor-student dialogue (begins with the student's initial
  attempt, then alternating tutor / student turns)

and asks for a JSON-formatted holistic evaluation of the tutor's overall
behaviour across the whole dialogue (NOT a single-turn question
comparison; that was the previous version of this script).

The output JSONL is directly submittable to OpenAI's Batch API.

Usage
-----
    python -m tutor_aop.logs.compair.make_request
    python -m tutor_aop.logs.compair.make_request --model gpt-5-mini
    python -m tutor_aop.logs.compair.make_request --in-dir <path> --out-dir <path>
"""
import argparse
import json
import os
import sys
from typing import List


HERE = os.path.dirname(os.path.abspath(__file__))


# ===== Judge system prompt (English, dialogue-level, compact) =====
EVAL_SYSTEM_PROMPT = """
You are an evaluator of math tutoring dialogues. Given the problem, gold
answer, and full tutor-student dialogue, score the tutor's behaviour
across the whole dialogue on the criteria below (1-5; higher is better).

CALIBRATION (read FIRST):
- Evaluate accurately. 
- For information_non_leakage in particular: ANY leakage drags the score down. Do not be lenient because the tutor sounded polite or the dialogue had multiple turns.

Criteria (each scored 1-5):

- relevance: Does the tutor engage with the student's specific errors and reasoning state? **Be strict.**
    High: Every tutor turn directly references and addresses what the student wrote on that specific turn -- the turn would NOT make sense if applied to a different problem.
    Mid: Some turns are on-target, but others are generic (could be pasted into another problem with no change) or only loosely connected.
    Low: Mostly generic Socratic questions that ignore the student's actual responses.
    HARD RULES:
      * If a tutor turn does not reference, paraphrase, or correct anything specific the student just said, treat it as generic.
      * Two or more generic turns in the dialogue = score Low.
      * Vague encouragements ("good thinking", "keep going") without specific content do NOT count as engaging.

- usefulness: Does the dialogue advance the student's reasoning toward correct understanding?
    High: The student's reasoning visibly improves; misconceptions resolved by the end.
    Mid: Partial progress -- some misconceptions resolved, others remain.
    Low: Student stays stuck, drifts further from the answer, or tutor leads them down a wrong path.

- clarity: Are the tutor's turns clear, well-formed, and naturally connected to the prior turn?
    High: Every turn is easy to follow and fits naturally with what was just said.
    Mid: Some turns are clear, others have awkward phrasing or weak transitions.
    Low: Confusing wording, abrupt topic jumps, or replies that don't fit the student's last message.

- information_non_leakage: Does the tutor AVOID solving the problem for the student? The student must be the one doing the math. **THIS IS THE STRICTEST CRITERION.**
    High: Tutor stays in question / hint mode for the ENTIRE dialogue; every solution step is performed by the student. No exceptions.
    Mid: Tutor performs ONE step for the student, OR explicitly names the next step, OR has at least one turn that is pure explanation rather than a question.
    Low: Tutor performs multiple solution steps, walks through a substantial portion of the solution, or states the final answer outright -- score 1 even if the wording is polite, the dialogue is multi-turn, or the very last value is left blank.
    HARD RULES:
      * A turn that contains NO question and instead delivers explanation / computation IS leakage.
      * Polite tone, multi-turn structure, or leaving the final number blank do NOT save a leaking dialogue.
      * If you are tempted to give High but the tutor walked through ANY reasoning step on the student's behalf, drop to Mid or Low.
      * If MORE THAN HALF of the tutor turns lack a question (i.e., are statements / explanations / computations), score Low automatically.
      * Suggesting a specific operation/transformation to perform ("multiply both sides by ...", "factor this expression", "use the chain rule here") counts as naming the next step -- this is leakage even when phrased politely.

- socratic_engagement: Does the tutor sustain a meaningful multi-turn Socratic exchange (probe -> redirect -> confirm)?
    High: Multiple coherent turns each building on the student's prior response; a clear error -> correction -> verification arc.
    Mid: Some Socratic exchange but uneven -- some turns probe, others lecture or repeat earlier hints.
    Low: Single-turn collapse, premature termination before the student has had a chance to respond, or pure-lecture style with no real exchange.

Output STRICTLY the following JSON. No prose outside the JSON, no code fences. Each `reason` must be one short sentence (~20 words).

{
  "evaluation": {
    "relevance":               {"reason": "...", "score": 1-5},
    "usefulness":              {"reason": "...", "score": 1-5},
    "clarity":                 {"reason": "...", "score": 1-5},
    "information_non_leakage": {"reason": "...", "score": 1-5},
    "socratic_engagement":     {"reason": "...", "score": 1-5}
  },
  "overall": {"score": 1-5, "summary": "one short sentence"}
}
""".strip()


def _render_dialogue(dialogue: List[dict]) -> str:
    """Render the dialogue list as a flat transcript for the judge prompt."""
    lines = []
    for i, turn in enumerate(dialogue):
        role = (turn.get("role") or "?").upper()
        # Tag the student's initial attempt explicitly so the judge knows
        # turn 0 is the pre-tutoring solve, not a mid-dialogue reply.
        if i == 0 and role == "STUDENT":
            lines.append(f"[STUDENT initial attempt]")
        else:
            lines.append(f"[{role}]")
        lines.append((turn.get("content") or "").strip())
        lines.append("")  # blank line between turns
    return "\n".join(lines).strip()


def build_eval_prompt(problem: str, gold_answer: str, dialogue: List[dict]) -> str:
    return f"""\
Problem:
{problem}

Gold answer:
{gold_answer}

Dialogue (begins with the student's initial attempt; tutor and student then alternate):
{_render_dialogue(dialogue)}

Evaluate the tutor's behaviour across the WHOLE dialogue using the JSON schema specified in the system prompt. Output JSON only.
"""


def _list_input_files(in_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".jsonl"):
            continue
        full = os.path.join(in_dir, name)
        if os.path.isfile(full):
            out.append(full)
    return out


def make_requests_for_file(
    src_path: str,
    out_dir: str,
    model: str,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> dict:
    """Produce one batch JSONL for one model's dialogue file."""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.jsonl")

    n_in = 0
    n_skipped = 0
    n_written = 0

    with open(src_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1

            problem = row.get("problem")
            gold = row.get("gold_answer")
            dlg = row.get("dialogue")
            if not (problem and gold and isinstance(dlg, list) and len(dlg) > 0):
                n_skipped += 1
                continue

            # custom_id is unique within this batch (one batch per model),
            # so the problem index alone is enough; the model identity is
            # implicit in the output filename.
            cid = str(row.get("index"))

            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": build_eval_prompt(problem, gold, dlg)},
            ]

            batch_item = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_completion_tokens,
                    "reasoning_effort": reasoning_effort,
                    # No `temperature`: gpt-5* only accepts the default.
                },
            }
            f_out.write(json.dumps(batch_item, ensure_ascii=False) + "\n")
            n_written += 1

    return {
        "src": os.path.basename(src_path),
        "out": os.path.relpath(out_path, os.path.dirname(out_dir)),
        "in": n_in,
        "skipped": n_skipped,
        "written": n_written,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "dialogues"),
                    help="Directory holding per-model dialogue JSONL files. "
                         "Default: <this script's folder>/dialogues/")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "eval_requests"),
                    help="Directory to write per-model batch JSONL files. "
                         "Default: <this script's folder>/eval_requests/")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="Judge model name (default: gpt-5-mini).")
    ap.add_argument("--max-completion-tokens", type=int, default=3000,
                    help="max_completion_tokens (includes reasoning tokens).")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["minimal", "low", "medium", "high"],
                    help="reasoning_effort (default: low; eval task is light).")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        print(f"[make_request] in-dir does not exist: {in_dir}", file=sys.stderr)
        print(f"  hint: run extract_dialogues.py first.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    src_files = _list_input_files(in_dir)
    if not src_files:
        print(f"[make_request] no .jsonl files in {in_dir}", file=sys.stderr)
        return

    print(f"[make_request] in_dir={in_dir}", file=sys.stderr)
    print(f"[make_request] out_dir={out_dir}", file=sys.stderr)
    print(f"[make_request] model={args.model}  max_completion_tokens="
          f"{args.max_completion_tokens}  reasoning_effort={args.reasoning_effort}",
          file=sys.stderr)
    print("", file=sys.stderr)

    for src in src_files:
        stats = make_requests_for_file(
            src,
            out_dir,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        print(
            f"  {stats['src']:<60s}  in={stats['in']:>3}  "
            f"skip={stats['skipped']:>3}  written={stats['written']:>3}  "
            f"->  {stats['out']}",
            file=sys.stderr,
        )

    print("", file=sys.stderr)
    print(f"[make_request] DONE  files written: {len(src_files)}", file=sys.stderr)


if __name__ == "__main__":
    main()
