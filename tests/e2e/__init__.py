"""End-to-end tests for agentique.

E2E tests validate the full stack against a live Docker Compose environment:
- MCP client → agentique bridge → A2A agents
- Real Redis and DynamoDB Local
- Complete protocol translation flows
"""
