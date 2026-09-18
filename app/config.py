"""Runtime settings (env) and scoring configuration (YAML)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    hf_token: str = ""
    hf_model: str = "Qwen/Qwen2.5-72B-Instruct"
    hf_provider: str = "auto"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_timeout_s: float = 90
    llm_max_retries: int = 3
    scoring_config: str = "config/scoring.yaml"
    data_dir: str = "data"


class Band(BaseModel):
    label: str
    min: float


class Grounding(BaseModel):
    ungrounded_level_penalty: int = Field(1, ge=0, le=4)
    min_quote_similarity: float = Field(0.8, ge=0, le=1)


class Knockout(BaseModel):
    enabled: bool = True
    cap: float = Field(45, ge=0, le=100)


class Limits(BaseModel):
    max_criteria: int = Field(12, ge=1, le=30)
    resume_chunk_tokens: int = Field(3000, ge=500)
    max_resume_chars: int = 60000
    max_jd_chars: int = 20000
    min_resume_chars: int = 200


class ScoringConfig(BaseModel):
    scorer_input_format: Literal["markdown", "json"] = "markdown"
    scorer_temperature: float = Field(0.0, ge=0, le=1.5)
    scorer_samples: int = Field(1, ge=1, le=7)
    importance_weights: dict[str, float]
    category_weights: dict[str, float]
    level_points: dict[int, float]
    grounding: Grounding = Grounding()
    knockout: Knockout = Knockout()
    recommendation_bands: list[Band]
    limits: Limits = Limits()

    @model_validator(mode="after")
    def _check(self) -> "ScoringConfig":
        if sorted(self.level_points) != [0, 1, 2, 3, 4]:
            raise ValueError("level_points must define levels 0..4")
        if any(w < 0 for w in [*self.importance_weights.values(), *self.category_weights.values()]):
            raise ValueError("weights must be non-negative")
        self.recommendation_bands.sort(key=lambda b: b.min, reverse=True)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_scoring_config() -> ScoringConfig:
    path = Path(get_settings().scoring_config)
    if not path.is_absolute():
        path = ROOT / path
    return ScoringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
