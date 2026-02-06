"""Configuration management using ``pydantic-settings``.

All settings are sourced from environment variables prefixed with
``AGENTIQUE_``, ``.env`` files, and constructor arguments. Types are
validated automatically by Pydantic.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


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
    enable_fastmcp_middleware: bool = True
    enable_resources_as_tools: bool = False
    enable_prompts_as_tools: bool = False

    # Performance
    cache_ttl: float = 300.0
    prefetch_cards: bool = True
    default_timeout: float = 60.0
    background_task_poll_interval_seconds: float = 2.0

    # FastMCP component transforms
    component_namespace: str = ""
    disabled_component_names: str = ""
    tool_transformations: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # A2A client transport and extension settings
    a2a_supported_transports: str = ""
    a2a_use_client_preference: bool = False
    a2a_extensions: str = ""
    a2a_card_path: str = ""
    a2a_push_notification_url: str = ""
    a2a_push_notification_token: str = ""
    a2a_push_notification_id: str = ""
    a2a_push_notification_auth: str = ""

    # Agent routing (parsed from env var)
    agents: str = ""  # raw env string; parsed by __main__

    @property
    def parsed_a2a_supported_transports(self) -> list[str]:
        return list(_split_csv(self.a2a_supported_transports))

    @property
    def parsed_a2a_extensions(self) -> list[str]:
        return list(_split_csv(self.a2a_extensions))

    @property
    def parsed_disabled_component_names(self) -> set[str]:
        return set(_split_csv(self.disabled_component_names))

    @property
    def background_task_poll_interval(self) -> timedelta:
        seconds = max(self.background_task_poll_interval_seconds, 0.1)
        return timedelta(seconds=seconds)
