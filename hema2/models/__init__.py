"""Model adapters used by the HEMA-2 evaluation harness.

The core memory implementation remains dependency-free.  These adapters use
only the Python standard library so the same runner can compare closed frontier
APIs and local Ollama models without provider SDK lock-in.
"""

from hema2.models.base import ChatModel, ModelRequest, ModelResponse
from hema2.models.registry import create_model, known_profiles

__all__ = [
    "ChatModel",
    "ModelRequest",
    "ModelResponse",
    "create_model",
    "known_profiles",
]
