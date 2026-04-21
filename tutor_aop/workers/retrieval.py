"""Retrieval worker.

Uses the tutor LLM to formulate a short search query, then ranks entries
in a JSONL pool by keyword-overlap with the query. No embedding dependency.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict

from ..utils import render_dialogue
from ..prompts.worker_prompts import RETRIEVAL_QUERY_SYSTEM, RETRIEVAL_QUERY_USER


_TOKEN_RE = re.compile(r"[A-Za-z0-9\\^]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _score(query_tokens: List[str], entry: Dict) -> float:
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    title_tokens = set(_tokenize(entry.get("title", "")))
    statement_tokens = set(_tokenize(entry.get("statement", "")))
    kw_tokens = set()
    for kw in entry.get("keywords", []):
        kw_tokens.update(_tokenize(kw))
    score = 0.0
    score += 3.0 * len(qset & title_tokens)
    score += 2.5 * len(qset & kw_tokens)
    score += 1.0 * len(qset & statement_tokens)
    return score


class RetrievalWorker:
    name = "retrieval"

    def __init__(
        self,
        client,
        pool_path: str,
        top_k: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        self.client = client
        self.pool_path = pool_path
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.entries = self._load_pool(pool_path)

    def _load_pool(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _formulate_query(self, problem: str, diagnosis: dict, dialogue: list) -> str:
        diag_str = json.dumps(diagnosis, ensure_ascii=False) if diagnosis else "(none)"
        messages = [
            {"role": "system", "content": RETRIEVAL_QUERY_SYSTEM},
            {"role": "user", "content": RETRIEVAL_QUERY_USER.format(
                problem=problem,
                diagnosis=diag_str,
                dialogue=render_dialogue(dialogue) or "(empty)",
            )},
        ]
        raw = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        q = (raw or "").strip().splitlines()[0] if raw else ""
        q = q.strip().strip('"').strip("'")
        if not q:
            q = problem[:60]
        return q

    def run(self, problem: str, diagnosis: dict, dialogue: list, subtask: str = "") -> dict:
        query = self._formulate_query(problem, diagnosis, dialogue)
        if not self.entries:
            return {"query": query, "retrieved_items": [], "note": "empty pool"}
        q_tokens = _tokenize(query)
        scored = [(_score(q_tokens, e), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for (s, e) in scored[: self.top_k] if s > 0]
        items = [
            {"id": e.get("id"), "title": e.get("title"), "statement": e.get("statement")}
            for e in top
        ]
        return {"query": query, "retrieved_items": items}
