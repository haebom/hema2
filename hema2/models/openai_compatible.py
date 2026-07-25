"""Minimal OpenAI-compatible chat completions adapter using urllib only."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from hema2.models.base import ModelRequest, ModelResponse


@dataclass
class OpenAICompatibleModel:
    model: str
    base_url: str
    api_key: str = ""
    provider: str = "openai-compatible"
    timeout_s: float = 120.0

    @classmethod
    def from_env(
        cls,
        model: str,
        *,
        provider: str,
        base_url_env: str,
        api_key_env: str,
        default_base_url: str,
    ) -> "OpenAICompatibleModel":
        return cls(
            model=model,
            provider=provider,
            base_url=os.getenv(base_url_env, default_base_url).rstrip("/"),
            api_key=os.getenv(api_key_env, ""),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {self.provider} at {endpoint}: {exc}") from exc

        latency = time.perf_counter() - started
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {self.provider} response shape: {raw}") from exc

        usage = raw.get("usage", {})
        return ModelResponse(
            text=text,
            model=raw.get("model", self.model),
            provider=self.provider,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=raw,
        )
