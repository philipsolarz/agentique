"""Root conftest for agentique test suite.

Provides:
- Automatic marker tagging based on directory structure
- Shared fixtures available across all test tiers
- Marker registration for pytest
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: Fast isolated unit tests (no external deps)",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests requiring running services or in-memory servers",
    )
    config.addinivalue_line(
        "markers",
        "e2e: Full end-to-end tests against Docker Compose stack",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests taking more than 5 seconds",
    )
