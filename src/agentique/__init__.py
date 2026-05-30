"""Agentique: an application-agnostic Python agent framework.

This is the package root. It intentionally re-exports nothing; the public surface
lives in the submodules — ``agentique.core`` (contracts + Runtime),
``agentique.tools``, ``agentique.memory``, ``agentique.testing``, the optional
``agentique.anthropic`` provider (behind the ``anthropic`` extra), and the
``agentique.code`` (harness) and ``agentique.console`` (application) layers.
"""
