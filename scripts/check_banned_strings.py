#!/usr/bin/env python3
"""CI guard: fail if any banned identifier appears in the repository.

The banned list is stored base64-encoded so this checker itself stays clean.
"""
from __future__ import annotations
import base64
import pathlib
import sys

_ENCODED = ["cmlzdQ==", "c3VwYW1lbW9yeQ==", "aHlwYW1lbW9yeQ==", "aHlwYXYy", "aHlwYXYz"]
BANNED = [base64.b64decode(s).decode() for s in _ENCODED]
SKIP_DIRS = {".git", "__pycache__", ".venv"}
SELF = pathlib.Path(__file__).resolve()


def main(root: str = ".") -> int:
    bad = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts) or p.resolve() == SELF:
            continue
        try:
            text = p.read_text(errors="ignore").lower()
        except (OSError, UnicodeDecodeError):
            continue
        for b in BANNED:
            if b in text:
                bad.append((str(p), b))
    for path, _ in bad:
        print(f"BANNED STRING FOUND: {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
