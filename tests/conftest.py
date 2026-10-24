from __future__ import annotations

from pathlib import Path

import pytest

from agent_verify.config import AppConfig, load_config


@pytest.fixture
def default_config() -> AppConfig:
    return load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
