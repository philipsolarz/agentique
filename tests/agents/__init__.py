"""Mock A2A agents for testing.

Provides deterministic agent implementations for testing:
- EchoAgent: Returns input verbatim
- StreamingAgent: Sends artifact chunks
- ErrorAgent: Immediately fails
- InputRequiredAgent: Requests additional input
- LongRunningAgent: Sends progress updates
"""
