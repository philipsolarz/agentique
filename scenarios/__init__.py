"""Throwaway exercise scripts that drive the real framework through the recording
layer and capture each run. Not part of the shipped ``agentique`` distribution and
never imported by it — these exist only to grade contracts against reality.
"""

from scenarios.driver import ScenarioRun, run_scenario

__all__ = ["ScenarioRun", "run_scenario"]
