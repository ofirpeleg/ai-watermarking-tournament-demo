"""
Config shared by generate.py and detect.py: env-var defaults.
No torch/transformers here either.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


@dataclass
class Settings:
    key: str
    max_tokens: int
    top_k: int
    entropy_min: float
    min_ratio: float
    rounds: int
    context_window: int
    p_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_settings() -> Settings:
    return Settings(
        key=_env_str("WATERMARK_KEY", "my-demo-secret-key"),
        max_tokens=_env_int("WATERMARK_MAX_TOKENS", 700),
        top_k=_env_int("WATERMARK_TOP_K", 20),
        entropy_min=_env_float("WATERMARK_ENTROPY_MIN", 2.0),
        min_ratio=_env_float("WATERMARK_MIN_RATIO", 0.3),
        rounds=_env_int("WATERMARK_ROUNDS", 30),
        context_window=_env_int("WATERMARK_CONTEXT", 4),
        p_threshold=_env_float("WATERMARK_P_THRESHOLD", 0.01),
    )
