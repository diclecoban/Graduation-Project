"""Small config helpers for YAML/JSON experiment files.

PyYAML is optional at import time so the existing pipeline remains usable even
before optional MLOps dependencies are installed. JSON config files always work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install PyYAML to read YAML config files: pip install pyyaml") from exc
        loaded = yaml.safe_load(path.read_text())
        return loaded or {}
    raise ValueError(f"Unsupported config extension: {path.suffix}")
