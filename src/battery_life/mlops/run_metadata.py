"""Lightweight run metadata utilities.

These helpers intentionally avoid mandatory external tracking dependencies.
They create local, versioned run folders under ``outputs/runs`` and can later be
connected to MLflow or WandB without changing the pipeline semantics.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_git(project_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_slug() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def create_run_dir(project_root: Path, label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")
    name = f"{run_slug()}_{safe_label}" if safe_label else run_slug()
    run_dir = project_root / "outputs" / "runs" / name
    (run_dir / "stages").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect_environment_metadata(project_root: Path, command: list[str] | None = None) -> dict[str, Any]:
    status = _safe_git(project_root, ["status", "--porcelain"]) or ""
    return {
        "created_at_utc": utc_timestamp(),
        "command": command or sys.argv,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _safe_git(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _safe_git(project_root, ["branch", "--show-current"]),
        "dirty_git_tree": bool(status),
        "git_status_porcelain": status.splitlines(),
    }


def write_pipeline_config(
    run_dir: Path,
    *,
    project_root: Path,
    selected_stages: list[str],
    phase: str | None,
    resume: bool,
    skip_download: bool,
) -> None:
    payload = {
        "run_type": "pipeline",
        "phase": phase,
        "selected_stages": selected_stages,
        "resume": resume,
        "skip_download": skip_download,
        "environment": collect_environment_metadata(project_root),
    }
    write_json(run_dir / "config.json", payload)


def _output_state(project_root: Path, outputs: list[Path]) -> list[dict[str, Any]]:
    states = []
    for out in outputs:
        try:
            rel = str(out.relative_to(project_root))
        except ValueError:
            rel = str(out)
        states.append({
            "path": rel,
            "exists": out.exists(),
            "is_dir": out.is_dir(),
            "size_bytes": out.stat().st_size if out.exists() and out.is_file() else None,
        })
    return states


def write_stage_metadata(
    run_dir: Path,
    *,
    project_root: Path,
    stage_name: str,
    description: str,
    command: list[str] | None,
    return_code: int,
    elapsed_seconds: float,
    outputs: list[Path],
) -> None:
    payload = {
        "stage": stage_name,
        "description": description,
        "command": command,
        "return_code": int(return_code),
        "status": "success" if return_code == 0 else "failed",
        "elapsed_seconds": float(elapsed_seconds),
        "finished_at_utc": utc_timestamp(),
        "outputs": _output_state(project_root, outputs),
    }
    write_json(run_dir / "stages" / f"{stage_name}.json", payload)
