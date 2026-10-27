#!/usr/bin/env python3
"""Check every import declared by a frozen BigCodeBench selection manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_manifest", type=Path)
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="Import each top-level module in an isolated child process.",
    )
    parser.add_argument(
        "--import-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-module timeout used with --import-check (default: 60).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    tasks = payload.get("task_records", [])
    libraries = sorted(
        {
            str(library)
            for task in tasks
            for library in (
                task.get("libraries")
                or task.get("public_task", {}).get("libraries")
                or []
            )
        }
    )
    top_level_modules = sorted({library.split(".")[0] for library in libraries})
    missing = [
        module
        for module in top_level_modules
        if importlib.util.find_spec(module) is None
    ]
    import_failures: list[dict[str, str]] = []
    if args.import_check and not missing:
        for module in top_level_modules:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        (
                            "import importlib; "
                            f"importlib.import_module({module!r})"
                        ),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.import_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                import_failures.append(
                    {
                        "module": module,
                        "error": (
                            "import timed out after "
                            f"{args.import_timeout_seconds:g} seconds"
                        ),
                    }
                )
                continue
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout).strip()
                import_failures.append(
                    {
                        "module": module,
                        "error": error[-2000:],
                    }
                )

    ready = not missing and not import_failures
    print(
        json.dumps(
            {
                "python": sys.executable,
                "tasks": len(tasks),
                "declared_libraries": len(top_level_modules),
                "available_libraries": len(top_level_modules) - len(missing),
                "missing_libraries": missing,
                "runtime_import_check": args.import_check,
                "import_failures": import_failures,
                "ready": ready,
            },
            indent=2,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
