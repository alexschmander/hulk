"""Manifest serialization and stable identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]
