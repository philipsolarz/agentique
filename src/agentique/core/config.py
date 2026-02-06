"""Configuration management using ``pydantic-settings``.

All settings are sourced from environment variables prefixed with
``AGENTIQUE_``, ``.env`` files, and constructor arguments. Types are
validated automatically by Pydantic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdapterConfig(BaseSettings):
    """Configuration for a single agent adapter."""

    model_config = SettingsConfigDict(extra="allow")

    protocol: str = "a2a"
    base_url: str = ""
    timeout: float = 60.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentiqueConfig(BaseSettings):
    """Top-level agentique server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIQUE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server identity
    name: str = "Agentique"

    # Transport
    transport: Literal["stdio", "streamable-http", "sse", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000

    # Feature flags
    enable_background_tasks: bool = True
    enable_elicitation: bool = True
    enable_tool_confirmation: bool = True

    # Performance
    cache_ttl: float = 300.0
    prefetch_cards: bool = True
    default_timeout: float = 60.0

    # Agent routing (parsed from env var)
    agents: str = ""  # raw env string; parsed by __main__
