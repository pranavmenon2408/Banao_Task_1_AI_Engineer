"""Runtime settings (env) and scoring configuration (YAML)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _alias(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    """Env settings. The text model (both agents) and the OCR vision model each get a provider, a model and optional
    fallback models, so either can run on Hugging Face, OpenAI, Gemini, Groq, Mistral, OpenRouter, Together, Ollama
    or any OpenAI-compatible endpoint (app/providers.py). Old HF_* names still work as aliases."""
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", populate_by_name=True)

    # text model: both agents
    llm_provider: str = "huggingface"
    llm_model: str = Field("meta-llama/Llama-3.3-70B-Instruct", validation_alias=_alias("LLM_MODEL", "HF_MODEL"))
    llm_fallback_models: str = ""       # comma-separated, tried in order when the main model is not available
    llm_api_key: str = ""               # optional; otherwise the provider's own variable (HF_TOKEN, OPENAI_API_KEY, ...)
    llm_endpoint: str = ""              # only for provider "openai-compatible" (or to override a provider's URL)
    # Hugging Face only: which inference provider serves the model ("auto" = first one enabled on the account)
    hf_inference_provider: str = Field("auto", validation_alias=_alias("HF_INFERENCE_PROVIDER", "HF_PROVIDER"))

    # vision model: OCR fallback for scanned resumes
    vlm_provider: str = ""              # empty = same as llm_provider
    vlm_model: str = Field("meta-llama/Llama-4-Scout-17B-16E-Instruct", validation_alias=_alias("VLM_MODEL", "HF_VLM_MODEL"))
    vlm_fallback_models: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    vlm_api_key: str = ""
    vlm_endpoint: str = ""
    vlm_hf_inference_provider: str = Field("auto", validation_alias=_alias("VLM_HF_INFERENCE_PROVIDER", "HF_VLM_PROVIDER"))

    tesseract_cmd: str = ""  # path to tesseract binary if it is not on PATH
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


class OcrConfig(BaseModel):
    enabled: bool = True
    max_pages: int = Field(4, ge=1, le=20)
    tesseract_dpi: int = Field(300, ge=72, le=600)
    tesseract_min_confidence: float = Field(70, ge=0, le=100)
    min_chars: int = Field(200, ge=0)
    vlm_fallback: bool = True
    vlm_max_side_px: int = Field(1600, ge=512, le=4096)
    vlm_max_parallel: int = Field(4, ge=1, le=16)


class ScoringConfig(BaseModel):
    scorer_input_format: Literal["markdown", "json"] = "markdown"
    scorer_temperature: float = Field(0.0, ge=0, le=1.5)
    scorer_samples: int = Field(1, ge=1, le=7)
    scoring_mode: Literal["single", "sectioned"] = "single"
    importance_weights: dict[str, float]
    category_weights: dict[str, float]
    level_points: dict[int, float]
    grounding: Grounding = Grounding()
    knockout: Knockout = Knockout()
    recommendation_bands: list[Band]
    limits: Limits = Limits()
    ocr: OcrConfig = OcrConfig()

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
