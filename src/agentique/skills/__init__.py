"""Agentique skills: pure, deterministic :class:`~agentique.core.skill.Skill`
implementations. Each is unit-testable in complete isolation and depends only on
``agentique-core`` contracts.
"""

from agentique.skills.extract_text import ExtractText

__all__ = ["ExtractText"]
