"""Instrument an Agent for capture — by composition, wrapping its Model and Tools.

Returns a wrapped *copy* (the original Agent is untouched), so a recorded run
produces an identical Result to an unrecorded one. Share one ``Recorder`` across
several agents — e.g. a parent and the sub-agents it dispatches — to interleave all
their seam crossings into a single ordered stream.
"""

from __future__ import annotations

from dataclasses import replace

from agentique.core import Agent
from observability.recorder import Recorder
from observability.wrappers import RecordingModel, RecordingTool


def instrument_agent(agent: Agent, recorder: Recorder) -> Agent:
    """Return a copy of ``agent`` whose Model and Tools record into ``recorder``."""
    return replace(
        agent,
        model=RecordingModel(agent.model, recorder),
        tools=tuple(RecordingTool(tool, recorder) for tool in agent.tools),
    )
