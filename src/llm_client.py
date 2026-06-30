"""
llm_client.py — Minimal OpenAI-compatible client for llama-server.

We don't depend on the `openai` package — it's a heavyweight dep with
heavy import time on Pi. A direct `requests` POST is enough for chat
completions and gives us full control over timeouts and retries.

llama-server API (when started with --port 8080):
  POST http://127.0.0.1:8080/v1/chat/completions
  Body (subset we use):
    {
      "model": "...",                  # ignored by llama-server
      "messages": [{"role":..., "content":...}, ...],
      "temperature": 0.1,
      "top_p": 0.9,
      "max_tokens": 220,
      "stream": false,
      "stop": ["\n\nQ:", "Context:"]   # optional stop sequences
    }
  Response: standard OpenAI ChatCompletion format.

To start llama-server on Pi 5:
  ./llama-server \
    --model qwen3-0.8b-instruct-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 2048 \
    --threads 4 \
    --n-gpu-layers 0          # Pi has no GPU, all CPU
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    text: str                # the assistant's message content
    finish_reason: str       # "stop" | "length" | "error"
    prompt_tokens: int       # 0 if llama-server doesn't report
    completion_tokens: int
    latency_s: float
    raw: Optional[Dict[str, Any]] = None  # raw response for debugging


class LLMClient:
    """OpenAI-compatible chat client for llama-server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        retries: Optional[int] = None,
        retry_backoff_ms: Optional[int] = None,
    ):
        cfg = get_config()["llm"]
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.api_key = api_key or cfg.get("api_key", "not-needed")
        self.model = model or cfg["model"]
        self.timeout_s = timeout_s if timeout_s is not None else cfg["timeout_s"]
        self.retries = retries if retries is not None else cfg.get("retries", 1)
        self.retry_backoff_ms = (
            retry_backoff_ms if retry_backoff_ms is not None else cfg.get("retry_backoff_ms", 250)
        )
        self._session = requests.Session()
        # llama-server doesn't validate the key, but OpenAI client convention:
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of {role, content} dicts. System + user is enough.
            temperature: Override config default.
            top_p: Override config default.
            max_tokens: Override config default.
            stop: Stop sequences.
            timeout_s: Per-attempt timeout. Falls back to self.timeout_s.

        Returns:
            LLMResponse with the assistant's text and metadata.
        """
        cfg = get_config()["llm"]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else cfg["temperature"],
            "top_p": top_p if top_p is not None else cfg["top_p"],
            "max_tokens": max_tokens if max_tokens is not None else cfg["max_tokens"],
            "stream": False,
        }
        if stop is not None:
            payload["stop"] = stop

        url = f"{self.base_url}/chat/completions"
        attempt_timeout = timeout_s if timeout_s is not None else self.timeout_s

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._session.post(url, json=payload, timeout=attempt_timeout)
                latency = time.perf_counter() - t0
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"llama-server HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"].strip()
                finish = choice.get("finish_reason", "stop")
                usage = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    finish_reason=finish,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=latency,
                    raw=data,
                )
            except (requests.RequestException, RuntimeError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                latency = time.perf_counter() - t0
                logger.warning(
                    "LLM call failed",
                    extra={
                        "attempt": attempt + 1,
                        "error": str(e)[:200],
                        "latency_s": round(latency, 2),
                    },
                )
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_ms / 1000.0)

        # All retries exhausted
        return LLMResponse(
            text="",
            finish_reason="error",
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=0.0,
            raw=None,
        )

    def health_check(self) -> bool:
        """Quick check that llama-server is reachable."""
        try:
            # llama-server exposes /health
            r = self._session.get(
                f"{self.base_url}/health", timeout=2.0
            )
            # /models is also a good endpoint
            if r.status_code != 200:
                r = self._session.get(
                    f"{self.base_url}/models", timeout=2.0
                )
            return r.status_code == 200
        except requests.RequestException:
            return False


# ---------- Singleton accessor ----------
_CLIENT: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
