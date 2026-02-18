"""Conftest for unit tests.

Unit tests are fast, isolated tests that don't require external services.
They may use mocks and in-memory implementations.
"""

import sys
from pathlib import Path

# Make the a2a_test_agent package importable for cogito agent tests
_a2a_src = Path(__file__).parent.parent.parent / "a2a_test_agent" / "src"
if str(_a2a_src) not in sys.path:
    sys.path.insert(0, str(_a2a_src))
