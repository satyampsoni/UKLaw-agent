"""
Configuration module for UK LawAssistant.

Uses pydantic-settings to load and validate configuration from environment
variables and .env files. Every config value is typed and validated at
startup — if something is missing or malformed, the app fails fast with
a clear error instead of silently misbehaving at runtime.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class RelaxAISettings(BaseSettings):
    """
    Relax AI API configuration.

    All fields map to environment variables prefixed with RELAX_AI_.
    Example: RELAX_AI_API_KEY -> api_key
    """

    model_config = SettingsConfigDict(
        env_prefix="RELAX_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(
        ...,  # Required — no default, must be provided
        description="Relax AI API key for authentication",
        min_length=1,
    )
    base_url: str = Field(
        default="https://api.relax.ai/v1",
        description="Base URL for Relax AI API",
    )
    model: str = Field(
        default="DeepSeek-V31-Terminus",
        description="Model identifier to use for inference",
    )
    max_tokens: int = Field(
        default=5000,
        ge=1,
        le=32768,
        description="Maximum tokens in LLM response",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low = deterministic (good for legal).",
    )
    timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="HTTP request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed API calls",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL doesn't have a trailing slash."""
        return v.rstrip("/")


class AppSettings(BaseSettings):
    """
    Application-level configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(
        default="development",
        description="Runtime environment: development, staging, production",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    app_name: str = Field(
        default="UK LawAssistant",
        description="Application name for logging and observability",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


class Settings(BaseSettings):
    """
    Root settings container.

    Composes all sub-settings into a single object.
    Access via: get_settings().relax_ai.api_key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    relax_ai: RelaxAISettings = Field(default_factory=RelaxAISettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Uses lru_cache so settings are loaded once from env/files
    and reused across the entire application. This is the only
    way any module should access configuration.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        print(settings.relax_ai.api_key)
    """
    return Settings()
