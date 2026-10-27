"""Append-only exact-compatibility cache used by the branching generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent_verify.logging import JsonlStore, canonical_json


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def compatibility_fingerprint(
    *,
    config_fingerprint: str,
    software_sha256: str,
    dataset_versions: dict[str, str],
    workflow_version: str = "fixed-planner-coder-reviewer-v1",
) -> str:
    return sha256_json(
        {
            "config_fingerprint": config_fingerprint,
            "software_sha256": software_sha256,
            "dataset_versions": dataset_versions,
            "workflow_version": workflow_version,
        }
    )


def cache_key(kind: str, compatibility: str, identity: dict[str, Any]) -> str:
    return f"{kind}_{sha256_json({'compatibility': compatibility, 'identity': identity})[:32]}"


class ExactCache:
    """Index immutable records and fail loudly on conflicting duplicate keys."""

    def __init__(self, path: str | Path, compatibility: str) -> None:
        self.store = JsonlStore(path)
        self.compatibility = compatibility
        self._index: dict[str, dict[str, Any]] = {}
        for row in self.store.read_all():
            key = str(row.get("cache_key", ""))
            if not key or row.get("compatibility_fingerprint") != compatibility:
                continue
            previous = self._index.get(key)
            if previous is not None and canonical_json(previous) != canonical_json(row):
                raise ValueError(f"Conflicting immutable cache records for {key}")
            self._index[key] = row

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._index.get(key)
        if row is None or row.get("status") != "completed":
            return None
        return row

    def put(self, row: dict[str, Any]) -> None:
        key = str(row["cache_key"])
        if row.get("compatibility_fingerprint") != self.compatibility:
            raise ValueError("Refusing to cache a record with a different compatibility fingerprint")
        previous = self._index.get(key)
        if previous is not None:
            if canonical_json(previous) != canonical_json(row):
                raise ValueError(f"Conflicting immutable cache write for {key}")
            return
        self.store.append(row)
        self._index[key] = row

    def completed_keys(self) -> set[str]:
        return {key for key, row in self._index.items() if row.get("status") == "completed"}
