"""Retrieval worker.

Uses the tutor LLM to formulate a short search query, then ranks entries
in a JSONL pool by keyword-overlap with the query. No embedding dependency.

Pool schema (new rag_pool.jsonl):
    {id, title, categories: [str], toplevel_categories: [str], statement, keywords: [str]}

The older proofwiki_examples.jsonl schema (`category` as a single string)
is still accepted transparently.
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


def _score(qset: set, entry: Dict) -> float:
    if not qset:
        return 0.0
    score = 0.0
    score += 3.0 * len(qset & entry["_title_tokens"])
    score += 2.5 * len(qset & entry["_kw_tokens"])
    score += 1.5 * len(qset & entry["_cat_tokens"])
    score += 1.0 * len(qset & entry["_statement_tokens"])
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
        out: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue

                title_tokens = set(_tokenize(e.get("title", "")))
                statement_tokens = set(_tokenize(e.get("statement", "")))
                kw_tokens: set = set()
                for kw in e.get("keywords", []) or []:
                    kw_tokens.update(_tokenize(kw))

                cat_tokens: set = set()
                # New schema: categories is a list.
                for cat in e.get("categories", []) or []:
                    cat_tokens.update(_tokenize(cat))
                # toplevel_categories are broad (Algebra/Analysis/...) and
                # match almost everything, so we skip them to avoid noise.
                # Old schema: single `category` string.
                if isinstance(e.get("category"), str):
                    cat_tokens.update(_tokenize(e["category"]))

                e["_title_tokens"] = title_tokens
                e["_statement_tokens"] = statement_tokens
                e["_kw_tokens"] = kw_tokens
                e["_cat_tokens"] = cat_tokens
                out.append(e)
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
        qset = set(_tokenize(query))
        scored = [(_score(qset, e), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for (s, e) in scored[: self.top_k] if s > 0]
        items = [
            {
                "id": e.get("id"),
                "title": e.get("title"),
                "categories": e.get("categories") or (
                    [e["category"]] if isinstance(e.get("category"), str) else []
                ),
                "statement": e.get("statement"),
            }
            for e in top
        ]
        return {"query": query, "retrieved_items": items}
