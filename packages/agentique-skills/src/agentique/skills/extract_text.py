"""A pure skill: collapse a message's content into its plain text.

Picks out the ``TextBlock`` content of a message and joins it, ignoring tool-use
and tool-result blocks. The Runtime uses this to derive the final assistant text
for a ``Completed`` result, but the skill itself knows nothing about runs — it is
a pure function of a message, testable in complete isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentique.core.messages import Message, TextBlock


@dataclass(frozen=True, slots=True)
class ExtractText:
    """Join the text blocks of a message with ``separator``.

    A frozen dataclass rather than a bare function so its one knob (the
    separator) is explicit configuration, and so it satisfies the
    ``Skill[Message, str]`` shape as a value that can be held by an Agent.
    """

    separator: str = ""

    def __call__(self, input: Message) -> str:
        return self.separator.join(
            block.text for block in input.content if isinstance(block, TextBlock)
        )
