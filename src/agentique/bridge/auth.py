"""Auth scheme parsing and typed credential elicitation helpers.

Provides utilities to parse ``SecurityScheme`` objects from A2A agent cards
into a normalised ``SecuritySchemeInfo`` dataclass, and to construct an
appropriate ``ctx.elicit()`` call (message + ``response_type``) for each
scheme family.

FastMCP 3.0.0b1 does not support ``mode="url"`` on ``ctx.elicit()``, so OAuth2
authorization URLs are embedded in the message text and we collect the resulting
authorization code via a typed Pydantic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Credential models (typed elicitation forms)
# ---------------------------------------------------------------------------


class ApiKeyCredentials(BaseModel):
    """Credentials for API-key authenticated agents."""

    api_key: str = Field(description="API key value")


class BearerCredentials(BaseModel):
    """Bearer / HTTP-auth token credentials."""

    token: str = Field(description="Bearer token")


class OAuthCodeCredentials(BaseModel):
    """OAuth2 authorization-code exchange credentials."""

    auth_code: str = Field(description="Authorization code from OAuth2 redirect")


# ---------------------------------------------------------------------------
# Normalised scheme descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecuritySchemeInfo:
    """Normalised representation of an A2A agent security scheme."""

    name: str
    """The scheme name as declared in the agent card's ``security_schemes``."""

    scheme_type: str
    """One of: ``"api_key"``, ``"http"``, ``"oauth2"``, ``"oidc"``, ``"mtls"``."""

    auth_url: str | None = None
    """Authorization endpoint (OAuth2 / OIDC) or ``None``."""

    token_url: str | None = None
    """Token endpoint (OAuth2 client credentials) or ``None``."""

    scopes: list[str] | None = None
    """List of required OAuth2 scopes, or ``None``."""

    header_name: str | None = None
    """HTTP header / query-param name for API-key schemes."""

    http_scheme: str | None = None
    """HTTP auth scheme (e.g. ``"bearer"``, ``"basic"``) or ``None``."""

    description: str | None = None
    """Human-readable description from the scheme definition."""


# ---------------------------------------------------------------------------
# Scheme parser
# ---------------------------------------------------------------------------


def parse_security_scheme(name: str, scheme: Any) -> SecuritySchemeInfo:
    """Convert an a2a-sdk ``SecurityScheme`` (or raw dict) into ``SecuritySchemeInfo``.

    Args:
        name: The scheme key from ``AgentCard.security_schemes``.
        scheme: An a2a-sdk ``SecurityScheme`` RootModel, a concrete scheme
            type, or a plain ``dict`` (for cards loaded outside the SDK).

    Returns:
        A normalised :class:`SecuritySchemeInfo`.
    """
    # Unwrap RootModel if necessary
    inner: Any = scheme
    if hasattr(scheme, "root"):
        inner = scheme.root

    # Plain-dict fallback
    if isinstance(inner, dict):
        return _parse_dict_scheme(name, inner)

    scheme_type_raw: str = getattr(inner, "type", "") or ""

    if scheme_type_raw == "apiKey":
        in_val = getattr(inner, "in_", None)
        in_str = in_val.value if hasattr(in_val, "value") else str(in_val) if in_val else None
        return SecuritySchemeInfo(
            name=name,
            scheme_type="api_key",
            header_name=getattr(inner, "name", None),
            description=getattr(inner, "description", None),
        )

    if scheme_type_raw == "http":
        raw_scheme = getattr(inner, "scheme", None) or ""
        return SecuritySchemeInfo(
            name=name,
            scheme_type="http",
            http_scheme=raw_scheme.lower() if raw_scheme else None,
            description=getattr(inner, "description", None),
        )

    if scheme_type_raw == "oauth2":
        flows = getattr(inner, "flows", None)
        auth_url: str | None = None
        token_url: str | None = None
        scopes: list[str] = []
        if flows is not None:
            # Authorization code takes priority; fall through to implicit, then CC
            ac = getattr(flows, "authorization_code", None)
            implicit = getattr(flows, "implicit", None)
            cc = getattr(flows, "client_credentials", None)
            if ac is not None:
                auth_url = getattr(ac, "authorization_url", None)
                token_url = getattr(ac, "token_url", None)
                raw_scopes = getattr(ac, "scopes", None)
                scopes = list(raw_scopes.keys()) if isinstance(raw_scopes, dict) else []
            elif implicit is not None:
                auth_url = getattr(implicit, "authorization_url", None)
                raw_scopes = getattr(implicit, "scopes", None)
                scopes = list(raw_scopes.keys()) if isinstance(raw_scopes, dict) else []
            elif cc is not None:
                token_url = getattr(cc, "token_url", None)
                raw_scopes = getattr(cc, "scopes", None)
                scopes = list(raw_scopes.keys()) if isinstance(raw_scopes, dict) else []
        return SecuritySchemeInfo(
            name=name,
            scheme_type="oauth2",
            auth_url=auth_url,
            token_url=token_url,
            scopes=scopes or None,
            description=getattr(inner, "description", None),
        )

    if scheme_type_raw == "openIdConnect":
        return SecuritySchemeInfo(
            name=name,
            scheme_type="oidc",
            auth_url=getattr(inner, "open_id_connect_url", None),
            description=getattr(inner, "description", None),
        )

    if scheme_type_raw == "mutualTLS":
        return SecuritySchemeInfo(
            name=name,
            scheme_type="mtls",
            description=getattr(inner, "description", None),
        )

    # Unknown / unrecognised — preserve raw type string
    return SecuritySchemeInfo(
        name=name,
        scheme_type=scheme_type_raw or "unknown",
        description=getattr(inner, "description", None),
    )


def _parse_dict_scheme(name: str, d: dict[str, Any]) -> SecuritySchemeInfo:
    """Parse a plain-dict security scheme (card loaded outside the SDK)."""
    t = d.get("type") or d.get("scheme_type") or "unknown"
    return SecuritySchemeInfo(
        name=name,
        scheme_type=_normalise_type(t),
        auth_url=d.get("auth_url") or d.get("authorizationUrl") or d.get("openIdConnectUrl"),
        token_url=d.get("token_url") or d.get("tokenUrl"),
        scopes=d.get("scopes"),
        header_name=d.get("header_name") or d.get("name"),
        http_scheme=d.get("http_scheme") or d.get("scheme"),
        description=d.get("description"),
    )


def _normalise_type(raw: str) -> str:
    mapping = {
        "apikey": "api_key",
        "http": "http",
        "oauth2": "oauth2",
        "openidconnect": "oidc",
        "mutualtls": "mtls",
    }
    return mapping.get(raw.lower(), raw)


# ---------------------------------------------------------------------------
# Elicitation selector
# ---------------------------------------------------------------------------

_PRIORITY = ["api_key", "http", "oauth2", "oidc", "mtls"]


def select_auth_elicitation(
    schemes: dict[str, "SecuritySchemeInfo"],
) -> tuple[str, type | None]:
    """Choose the best ``ctx.elicit()`` call for the given security schemes.

    Returns a ``(message_text, response_type)`` tuple suitable for passing
    directly to ``await ctx.elicit(message_text, response_type=response_type)``.

    Priority order: api_key → http → oauth2 (authorization_code / implicit) →
    oauth2 (client_credentials) → oidc → mtls.

    When *schemes* is empty or ``None``, returns the legacy generic prompt
    with ``response_type=None``.
    """
    if not schemes:
        return (
            "The agent requires authentication. Please provide your credentials.",
            None,
        )

    # Sort by priority
    ordered = sorted(
        schemes.values(),
        key=lambda s: _PRIORITY.index(s.scheme_type) if s.scheme_type in _PRIORITY else 99,
    )

    for info in ordered:
        result = _elicitation_for(info)
        if result is not None:
            return result

    return (
        "The agent requires authentication. Please provide your credentials.",
        None,
    )


def _elicitation_for(info: SecuritySchemeInfo) -> tuple[str, type | None] | None:
    if info.scheme_type == "api_key":
        header_hint = f" (sent as `{info.header_name}`)" if info.header_name else ""
        msg = f"The agent requires an API key{header_hint}."
        return msg, ApiKeyCredentials

    if info.scheme_type == "http":
        scheme = info.http_scheme or "bearer"
        if scheme == "bearer":
            return "The agent requires a Bearer token.", BearerCredentials
        if scheme == "basic":
            return "The agent requires Basic auth. Enter your pre-encoded token.", BearerCredentials
        return f"The agent requires HTTP auth (scheme: {scheme}).", BearerCredentials

    if info.scheme_type == "oauth2":
        if info.auth_url:
            scopes_hint = ""
            if info.scopes:
                scopes_hint = f"\nRequired scopes: {', '.join(info.scopes)}"
            msg = (
                f"The agent requires OAuth2 authorization.\n"
                f"Visit the following URL to authorize and obtain a code:\n"
                f"  {info.auth_url}{scopes_hint}\n\n"
                f"After authorizing, paste the authorization code below."
            )
            return msg, OAuthCodeCredentials
        # Client credentials or password: collect a pre-acquired token
        return (
            "The agent requires OAuth2 (client credentials). "
            "Enter your pre-acquired bearer token.",
            BearerCredentials,
        )

    if info.scheme_type == "oidc":
        url_hint = f"\nOIDC discovery URL: {info.auth_url}" if info.auth_url else ""
        msg = (
            f"The agent requires OpenID Connect authentication.{url_hint}\n"
            "After completing the OIDC flow, paste the authorization code below."
        )
        return msg, OAuthCodeCredentials

    if info.scheme_type == "mtls":
        return (
            "The agent requires mutual TLS (mTLS) authentication. "
            "mTLS cannot be handled via elicitation — "
            "configure client certificates at the transport layer.",
            None,
        )

    return None
