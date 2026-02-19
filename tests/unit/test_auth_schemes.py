"""Tests for agentique.bridge.auth — SecuritySchemeInfo, credential models,
parse_security_scheme(), and select_auth_elicitation()."""

from __future__ import annotations

import pytest

from agentique.bridge.auth import (
    ApiKeyCredentials,
    BearerCredentials,
    OAuthCodeCredentials,
    SecuritySchemeInfo,
    parse_security_scheme,
    select_auth_elicitation,
)


# ---------------------------------------------------------------------------
# parse_security_scheme — Pydantic a2a-sdk objects
# ---------------------------------------------------------------------------


def test_parse_api_key_scheme():
    from a2a.types import APIKeySecurityScheme, SecurityScheme

    scheme = SecurityScheme(root=APIKeySecurityScheme(type="apiKey", in_="header", name="X-API-Key"))
    info = parse_security_scheme("mykey", scheme)

    assert info.name == "mykey"
    assert info.scheme_type == "api_key"
    assert info.header_name == "X-API-Key"
    assert info.description is None


def test_parse_http_bearer_scheme():
    from a2a.types import HTTPAuthSecurityScheme, SecurityScheme

    scheme = SecurityScheme(root=HTTPAuthSecurityScheme(type="http", scheme="bearer"))
    info = parse_security_scheme("bearerAuth", scheme)

    assert info.scheme_type == "http"
    assert info.http_scheme == "bearer"


def test_parse_http_basic_scheme():
    from a2a.types import HTTPAuthSecurityScheme, SecurityScheme

    scheme = SecurityScheme(root=HTTPAuthSecurityScheme(type="http", scheme="basic"))
    info = parse_security_scheme("basicAuth", scheme)

    assert info.scheme_type == "http"
    assert info.http_scheme == "basic"


def test_parse_oauth2_authorization_code():
    from a2a.types import (
        AuthorizationCodeOAuthFlow,
        OAuthFlows,
        OAuth2SecurityScheme,
        SecurityScheme,
    )

    flows = OAuthFlows(
        authorization_code=AuthorizationCodeOAuthFlow(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            scopes={"read": "Read access", "write": "Write access"},
        )
    )
    scheme = SecurityScheme(root=OAuth2SecurityScheme(type="oauth2", flows=flows))
    info = parse_security_scheme("oauth2Auth", scheme)

    assert info.scheme_type == "oauth2"
    assert info.auth_url == "https://example.com/oauth/authorize"
    assert info.token_url == "https://example.com/oauth/token"
    assert info.scopes is not None
    assert "read" in info.scopes
    assert "write" in info.scopes


def test_parse_oauth2_implicit():
    from a2a.types import ImplicitOAuthFlow, OAuthFlows, OAuth2SecurityScheme, SecurityScheme

    flows = OAuthFlows(
        implicit=ImplicitOAuthFlow(
            authorization_url="https://example.com/oauth/implicit",
            scopes={"read": "Read access"},
        )
    )
    scheme = SecurityScheme(root=OAuth2SecurityScheme(type="oauth2", flows=flows))
    info = parse_security_scheme("implicitAuth", scheme)

    assert info.scheme_type == "oauth2"
    assert info.auth_url == "https://example.com/oauth/implicit"
    assert info.token_url is None


def test_parse_oauth2_client_credentials():
    from a2a.types import ClientCredentialsOAuthFlow, OAuthFlows, OAuth2SecurityScheme, SecurityScheme

    flows = OAuthFlows(
        client_credentials=ClientCredentialsOAuthFlow(
            token_url="https://example.com/token",
            scopes={},
        )
    )
    scheme = SecurityScheme(root=OAuth2SecurityScheme(type="oauth2", flows=flows))
    info = parse_security_scheme("ccAuth", scheme)

    assert info.scheme_type == "oauth2"
    assert info.token_url == "https://example.com/token"
    assert info.auth_url is None


def test_parse_oidc_scheme():
    from a2a.types import OpenIdConnectSecurityScheme, SecurityScheme

    scheme = SecurityScheme(
        root=OpenIdConnectSecurityScheme(
            type="openIdConnect",
            open_id_connect_url="https://example.com/.well-known/openid-configuration",
        )
    )
    info = parse_security_scheme("oidcAuth", scheme)

    assert info.scheme_type == "oidc"
    assert info.auth_url == "https://example.com/.well-known/openid-configuration"


def test_parse_mtls_scheme():
    from a2a.types import MutualTLSSecurityScheme, SecurityScheme

    scheme = SecurityScheme(root=MutualTLSSecurityScheme(type="mutualTLS"))
    info = parse_security_scheme("mtlsAuth", scheme)

    assert info.scheme_type == "mtls"
    assert info.auth_url is None


# ---------------------------------------------------------------------------
# parse_security_scheme — plain-dict fallback
# ---------------------------------------------------------------------------


def test_parse_scheme_dict_api_key_fallback():
    d = {"type": "apiKey", "name": "X-Custom-Key", "in": "header"}
    info = parse_security_scheme("dictKey", d)

    assert info.scheme_type == "api_key"
    assert info.header_name == "X-Custom-Key"


def test_parse_scheme_dict_oauth2_fallback():
    d = {
        "type": "oauth2",
        "auth_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
    }
    info = parse_security_scheme("dictOAuth", d)

    assert info.scheme_type == "oauth2"
    assert info.auth_url == "https://auth.example.com/authorize"
    assert info.token_url == "https://auth.example.com/token"


# ---------------------------------------------------------------------------
# select_auth_elicitation
# ---------------------------------------------------------------------------


def test_select_auth_elicitation_api_key():
    schemes = {
        "k": SecuritySchemeInfo(name="k", scheme_type="api_key", header_name="X-API-Key"),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is ApiKeyCredentials
    assert "X-API-Key" in msg
    assert "API key" in msg


def test_select_auth_elicitation_http_bearer():
    schemes = {
        "b": SecuritySchemeInfo(name="b", scheme_type="http", http_scheme="bearer"),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is BearerCredentials
    assert "Bearer" in msg


def test_select_auth_elicitation_http_basic():
    schemes = {
        "basic": SecuritySchemeInfo(name="basic", scheme_type="http", http_scheme="basic"),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is BearerCredentials
    assert "Basic" in msg


def test_select_auth_elicitation_oauth2_with_auth_url():
    schemes = {
        "o": SecuritySchemeInfo(
            name="o",
            scheme_type="oauth2",
            auth_url="https://example.com/oauth/authorize",
        ),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is OAuthCodeCredentials
    assert "https://example.com/oauth/authorize" in msg


def test_select_auth_elicitation_oauth2_client_credentials():
    schemes = {
        "cc": SecuritySchemeInfo(
            name="cc",
            scheme_type="oauth2",
            token_url="https://example.com/token",
        ),
    }
    msg, rtype = select_auth_elicitation(schemes)

    # client_credentials → collect pre-acquired bearer token
    assert rtype is BearerCredentials


def test_select_auth_elicitation_oidc():
    schemes = {
        "oidc": SecuritySchemeInfo(
            name="oidc",
            scheme_type="oidc",
            auth_url="https://example.com/.well-known/openid-configuration",
        ),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is OAuthCodeCredentials
    assert "https://example.com/.well-known/openid-configuration" in msg


def test_select_auth_elicitation_mtls_returns_none():
    schemes = {
        "mtls": SecuritySchemeInfo(name="mtls", scheme_type="mtls"),
    }
    msg, rtype = select_auth_elicitation(schemes)

    assert rtype is None
    assert "mTLS" in msg or "mutual TLS" in msg.lower()


def test_select_auth_elicitation_empty_schemes():
    msg, rtype = select_auth_elicitation({})

    assert rtype is None
    assert "authentication" in msg.lower()


def test_select_auth_elicitation_api_key_beats_http():
    """api_key has higher priority than http."""
    schemes = {
        "b": SecuritySchemeInfo(name="b", scheme_type="http", http_scheme="bearer"),
        "k": SecuritySchemeInfo(name="k", scheme_type="api_key", header_name="X-Key"),
    }
    _msg, rtype = select_auth_elicitation(schemes)
    assert rtype is ApiKeyCredentials


# ---------------------------------------------------------------------------
# Credential model serialisation
# ---------------------------------------------------------------------------


def test_api_key_credentials_model_dump():
    creds = ApiKeyCredentials(api_key="secret-key-123")
    d = creds.model_dump()
    assert d == {"api_key": "secret-key-123"}


def test_bearer_credentials_model_dump():
    creds = BearerCredentials(token="mytoken")
    d = creds.model_dump()
    assert d == {"token": "mytoken"}


def test_oauth_code_credentials_model_dump():
    creds = OAuthCodeCredentials(auth_code="code-xyz")
    d = creds.model_dump()
    assert d == {"auth_code": "code-xyz"}


# ---------------------------------------------------------------------------
# card_parser.extract_security_schemes
# ---------------------------------------------------------------------------


def test_extract_security_schemes_from_dict_card():
    from agentique.adapters.a2a.card_parser import A2ACardParser

    card = {
        "name": "test-agent",
        "security_schemes": {
            "apiKey": {"type": "apiKey", "name": "X-Key", "in": "header"}
        },
    }
    parser = A2ACardParser()
    schemes = parser.extract_security_schemes(card)

    assert "apiKey" in schemes
    assert schemes["apiKey"].scheme_type == "api_key"


def test_extract_security_schemes_from_pydantic_card():
    from a2a.types import AgentCard, AgentCapabilities, AgentSkill, APIKeySecurityScheme, SecurityScheme

    card = AgentCard(
        name="test",
        url="http://localhost:9000",
        version="1.0",
        description="Test agent",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[AgentSkill(id="s1", name="skill1", description="A skill", tags=[])],
        security_schemes={
            "bearerAuth": SecurityScheme(
                root=APIKeySecurityScheme(type="apiKey", in_="header", name="Authorization")
            )
        },
    )
    from agentique.adapters.a2a.card_parser import A2ACardParser
    parser = A2ACardParser()
    schemes = parser.extract_security_schemes(card)

    assert "bearerAuth" in schemes
    assert schemes["bearerAuth"].scheme_type == "api_key"


def test_extract_security_schemes_empty_card():
    from agentique.adapters.a2a.card_parser import A2ACardParser

    parser = A2ACardParser()
    assert parser.extract_security_schemes({}) == {}
    assert parser.extract_security_schemes(None) == {}


# ---------------------------------------------------------------------------
# AgentProvider.get_security_schemes
# ---------------------------------------------------------------------------


def test_provider_get_security_schemes_delegates_to_cache():
    """get_security_schemes() returns schemes from the cached card entry."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from agentique.bridge.provider import AgentProvider, _CardCache
    from agentique.core.types import AgentInfo

    info = AgentInfo(name="demo", base_url="http://demo:9000")
    adapter = MagicMock()
    adapter.get_agent_card = AsyncMock(return_value={"name": "demo"})
    adapter.close = AsyncMock()

    # Pre-populate cache with a known scheme
    expected_scheme = SecuritySchemeInfo(name="k", scheme_type="api_key", header_name="X-Key")
    cache_entry = _CardCache(
        card={"name": "demo"},
        tools=[],
        prompts=[],
        resources=[],
        security_schemes={"k": expected_scheme},
    )

    provider = AgentProvider([info], adapter, prefetch_cards=False)
    provider._cache["demo"] = cache_entry

    schemes = asyncio.run(provider.get_security_schemes("demo"))
    assert "k" in schemes
    assert schemes["k"].scheme_type == "api_key"


def test_provider_get_security_schemes_missing_agent():
    """Unknown agent name returns empty dict, not an error."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from agentique.bridge.provider import AgentProvider
    from agentique.core.types import AgentInfo

    info = AgentInfo(name="demo", base_url="http://demo:9000")
    adapter = MagicMock()
    adapter.get_agent_card = AsyncMock(return_value=None)
    adapter.close = AsyncMock()

    provider = AgentProvider([info], adapter, prefetch_cards=False)
    result = asyncio.run(provider.get_security_schemes("nonexistent"))
    assert result == {}


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_security_scheme_info_exported_from_agentique():
    import agentique

    assert hasattr(agentique, "SecuritySchemeInfo")
    assert hasattr(agentique, "ApiKeyCredentials")
    assert hasattr(agentique, "BearerCredentials")
    assert hasattr(agentique, "OAuthCodeCredentials")
    assert hasattr(agentique, "parse_security_scheme")
    assert hasattr(agentique, "select_auth_elicitation")
