"""OpenAI-compatible client wrapper for vLLM servers.

Supports two separate endpoints (tutor, student) and a mock mode for
dry-run testing without live servers.
"""
from __future__ import annotations

import time
import random
from typing import List, Dict, Optional


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        retries: int = 5,
        mock: bool = False,
        mock_handler=None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.mock = mock
        self.mock_handler = mock_handler
        self._client = None

        if not mock:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        if self.mock:
            return self._mock_call(messages, temperature, max_tokens, stop)

        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("Empty response content")
                return content
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 10) + random.random()
                time.sleep(sleep_s)
        raise RuntimeError(f"LLM chat failed after {self.retries} retries: {last_err}")

    def _mock_call(self, messages, temperature, max_tokens, stop):
        if self.mock_handler is not None:
            return self.mock_handler(self.model, messages, temperature, max_tokens, stop)
        return "OK"


def build_clients_from_config(cfg: dict):
    """Build tutor and student clients from a config dict."""
    mock_enabled = cfg.get("mock", {}).get("enabled", False)
    mock_handler = None
    if mock_enabled:
        from .mock_backend import default_mock_handler
        mock_handler = default_mock_handler

    exp = cfg["experiment"]
    tutor_cfg = cfg["tutor_server"]
    student_cfg = cfg["student_server"]

    tutor = LLMClient(
        base_url=tutor_cfg["base_url"],
        model=tutor_cfg["model"],
        api_key=tutor_cfg.get("api_key", "EMPTY"),
        timeout=exp.get("request_timeout", 120),
        retries=exp.get("retry", 5),
        mock=mock_enabled,
        mock_handler=mock_handler,
    )
    student = LLMClient(
        base_url=student_cfg["base_url"],
        model=student_cfg["model"],
        api_key=student_cfg.get("api_key", "EMPTY"),
        timeout=exp.get("request_timeout", 120),
        retries=exp.get("retry", 5),
        mock=mock_enabled,
        mock_handler=mock_handler,
    )
    return tutor, student
