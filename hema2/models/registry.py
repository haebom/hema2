"""Configuration-driven model registry.

Profile names are stable experiment labels. Provider model IDs are intentionally
overridable because frontier APIs rename previews and aliases frequently.
"""
from __future__ import annotations

import os

from hema2.models.ollama import OllamaModel
from hema2.models.openai_compatible import OpenAICompatibleModel


_PROFILES = {
    "gpt-5.6": {
        "kind": "openai",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-5.6",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
    },
    "fable-5": {
        "kind": "compatible",
        "model_env": "FABLE_MODEL",
        "default_model": "fable-5",
        "base_url_env": "FABLE_BASE_URL",
        "api_key_env": "FABLE_API_KEY",
        "default_base_url": "http://localhost:8000/v1",
    },
    "kimi-k3": {
        "kind": "compatible",
        "model_env": "KIMI_MODEL",
        "default_model": "kimi-k3",
        "base_url_env": "KIMI_BASE_URL",
        "api_key_env": "KIMI_API_KEY",
        "default_base_url": "https://api.moonshot.ai/v1",
    },
    "ollama": {
        "kind": "ollama",
        "model_env": "OLLAMA_MODEL",
        "default_model": "qwen3:8b",
    },
}


def known_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def create_model(profile: str, model_override: str | None = None):
    if profile not in _PROFILES:
        raise ValueError(f"Unknown model profile '{profile}'. Choose from: {', '.join(known_profiles())}")
    cfg = _PROFILES[profile]
    model = model_override or os.getenv(cfg["model_env"], cfg["default_model"])
    if cfg["kind"] == "ollama":
        return OllamaModel.from_env(model)
    return OpenAICompatibleModel.from_env(
        model,
        provider=profile,
        base_url_env=cfg["base_url_env"],
        api_key_env=cfg["api_key_env"],
        default_base_url=cfg["default_base_url"],
    )
