"""Utility functions: JSON parsing with fallbacks, boxed extraction, answer equivalence."""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json_block(text: str) -> Optional[str]:
    """Pull a JSON-looking block out of free text.

    Handles: fenced ```json ... ```, fenced ``` ... ```, raw {...} / [...]
    """
    if text is None:
        return None

    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [s for s in (start_obj, start_arr) if s != -1]
    if not starts:
        return None
    start = min(starts)

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
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
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Try hard to parse JSON out of an LLM response."""
    if text is None:
        return default
    try:
        return json.loads(text)
    except Exception:
        pass
    block = extract_json_block(text)
    if block is None:
        return default
    try:
        return json.loads(block)
    except Exception:
        block2 = block.replace("\n", " ").replace("'", '"')
        try:
            return json.loads(block2)
        except Exception:
            return default


BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content (handles one level of nested braces)."""
    if text is None:
        return None
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        if inner.count("(") == inner.count(")"):
            s = inner
    return s


def answers_equivalent(pred: str, gold: str) -> bool:
    """Compare two answer strings, using sympy when possible.

    Returns True when pred is equivalent to gold.
    """
    if pred is None or gold is None:
        return False
    p = _normalize_latex(pred)
    g = _normalize_latex(gold)
    if p == g:
        return True
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify

        pe = parse_latex(p)
        ge = parse_latex(g)
        if pe == ge:
            return True
        try:
            if simplify(pe - ge) == 0:
                return True
        except Exception:
            pass
    except Exception:
        pass
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        pass
    return False


def render_dialogue(dialogue: list) -> str:
    """Format a dialogue list [{'role': 'tutor'|'student', 'content': ...}] as text."""
    lines = []
    for m in dialogue:
        role = m.get("role", "?").upper()
        lines.append(f"[{role}] {m.get('content', '')}")
    return "\n".join(lines)


def contains_end_signal(text: str) -> bool:
    if text is None:
        return False
    return "<end_of_conversation>" in text.lower() or "<end_of_conversation>" in text
