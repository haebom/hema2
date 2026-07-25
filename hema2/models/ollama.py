"""Ollama native chat adapter using the local /api/chat endpoint."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from hema2.models.base import ModelRequest, ModelResponse


@dataclass
class OllamaModel:
    model: str
    base_url: str = "http://localhost:11434"
    provider: str = "ollama"
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls, model: str) -> "OllamaModel":
        return cls(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))

    def complete(self, request: ModelRequest) -> ModelResponse:
        options: dict = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed

        payload = {
            "model": self.model,
            "messages": request.messages,
            "stream": False,
            "options": options,
        }
        endpoint = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {endpoint}. Start Ollama and pull model '{self.model}'."
            ) from exc

        latency = time.perf_counter() - started
        text = raw.get("message", {}).get("content")
        if text is None:
            raise RuntimeError(f"Unexpected Ollama response shape: {raw}")

        return ModelResponse(
            text=text,
            model=raw.get("model", self.model),
            provider=self.provider,
            latency_s=latency,
            prompt_tokens=raw.get("prompt_eval_count"),
            completion_tokens=raw.get("eval_count"),
            raw=raw,
        )
