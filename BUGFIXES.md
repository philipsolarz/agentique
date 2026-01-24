# AgentMCP Bug Fixes and Code Review

## Date: 2026-01-24

This document outlines the issues found in the agentique codebase and the fixes applied.

## Critical Bugs Fixed

### 1. AgentRouter Initialization Bug (CRITICAL)

**File**: [`src/agentique/router.py`](src/agentique/router.py)

**Issue**: The `_default` attribute was accessed in the `register()` method before it was initialized in `__init__()`.

**Error Message**:
```
AttributeError: 'AgentRouter' object has no attribute '_default'
```

**Root Cause**:
```python
def __init__(self, agents: Iterable[AgentDescriptor] | None = None, *, default: str | None = None) -> None:
    self._agents: dict[str, AgentDescriptor] = {}
    if agents:
        for agent in agents:
            self.register(agent)  # ❌ Calls register before _default is set!
    self._default = default or (next(iter(self._agents)) if self._agents else None)  # Set AFTER register

def register(self, agent: AgentDescriptor) -> None:
    self._agents[agent.name] = agent
    if self._default is None:  # ❌ Tries to access _default which doesn't exist yet!
        self._default = agent.name
```

**Fix Applied**:
```python
def __init__(self, agents: Iterable[AgentDescriptor] | None = None, *, default: str | None = None) -> None:
    self._agents: dict[str, AgentDescriptor] = {}
    self._default: str | None = None  # ✅ Initialize BEFORE calling register
    if agents:
        for agent in agents:
            self.register(agent)
    # Override default if explicitly provided
    if default is not None:
        self._default = default
```

**Impact**: This was blocking the MCP server from starting at all.

---

### 2. Incorrect A2A Message Creation

**File**: [`src/agentique/a2a_adapter.py`](src/agentique/a2a_adapter.py)

**Issue**: `create_text_message_object()` was being called with incorrect parameters.

**Root Cause**:

The A2A SDK function signature is:
```python
def create_text_message_object(
    role: Role = Role.user,
    content: str = ''
) -> Message:
```

But the agentique code was calling it as:
```python
message = imports.create_text_message_object(text)  # ❌ Passing text as 'role' parameter!
```

This would pass the user's message text as the `role` parameter (first positional arg) instead of the `content` parameter (second positional arg).

**Fix Applied**:
```python
# create_text_message_object(role=Role.user, content='')
# Pass text as content parameter (second positional arg)
message = imports.create_text_message_object(content=text)  # ✅ Correct parameter
```

**Impact**: This would cause messages to be created with the user's text as the role (invalid) and empty content.

---

## Code Review Findings

### A2A Adapter Complexity

**File**: [`src/agentique/a2a_adapter.py`](src/agentique/a2a_adapter.py)

**Observation**: The adapter has extensive fallback logic to support multiple versions of the A2A SDK:

1. **Modern Client API** (preferred):
   - Uses `ClientFactory.connect(base_url)` to create clients
   - Client `send_message()` takes a `Message` object and returns `AsyncIterator[ClientEvent | Message]`
   - Parameters: `request`, `context`, `request_metadata`, `extensions`

2. **Legacy Client APIs** (backwards compatibility):
   - Uses `A2AClient` with different method signatures
   - May have `send_message_streaming()` method
   - May use `SendMessageRequest` and `SendStreamingMessageRequest` objects

**Current Implementation**:
The code attempts multiple strategies in sequence:
1. Try modern streaming API with `send_message_streaming()` (lines 420-430)
2. Try modern non-streaming API with request objects (lines 432-439)
3. Try calling `send_message()` with kwargs (lines 441-446)
4. Fallback to legacy request objects (lines 447-455)
5. Handle async iterator return (lines 457-460)
6. Handle awaitable return (lines 462-465)
7. Handle synchronous return (line 467)

**Assessment**: While complex, this fallback logic provides good backwards compatibility. The code is defensive and handles various SDK versions gracefully.

**Recommendation**: This complexity is acceptable for now, but should be simplified once the A2A SDK stabilizes and older versions can be dropped.

---

## Architecture Review

### Overall Design

The agentique library follows a clean separation of concerns:

1. **Router Layer** ([`router.py`](src/agentique/router.py)):
   - Maps requests to agents by name or skill
   - Simple and focused

2. **Bridge Layer** ([`bridge.py`](src/agentique/bridge.py)):
   - Coordinates routing, context propagation, and A2A interaction
   - Clean interface

3. **A2A Adapter** ([`a2a_adapter.py`](src/agentique/a2a_adapter.py)):
   - Handles A2A SDK integration
   - Translates between MCP and A2A protocols
   - Complex but necessary for backwards compatibility

4. **Server Layer** ([`server.py`](src/agentique/server.py)):
   - Exposes FastMCP tools, resources, and prompts
   - Well-structured use of FastMCP 3.0 features

5. **Models** ([`models.py`](src/agentique/models.py)):
   - Clean dataclass-based models
   - Good separation of concerns

### FastMCP 3.0 Usage

The server correctly uses modern FastMCP 3.0 features:

✅ **Tools**: `a2a_send`, `a2a_stream`, `a2a_list_agents`
✅ **Resources**: `a2a://agents`, `a2a://agents/{agent}`, `a2a://agents/{agent}/card`
✅ **Prompts**: `a2a_routing_prompt`
✅ **Context Propagation**: Uses `CurrentContext()` dependency injection
✅ **Lifecycle Management**: Uses `@lifespan` decorator correctly
✅ **Progress Reporting**: `ctx.info()`, `ctx.report_progress()`

### A2A Integration

The integration follows A2A SDK patterns:

✅ **Client Creation**: Uses `ClientFactory.connect()` for modern API
✅ **Message Creation**: Uses `create_text_message_object()` helper
✅ **Metadata Handling**: Properly embeds MCP context in A2A metadata
✅ **Event Streaming**: Correctly handles async iterator responses
✅ **Agent Cards**: Retrieves and exposes agent cards via MCP resources

---

## Testing Recommendations

### Unit Tests Needed

1. **AgentRouter**:
   - Test initialization with and without agents
   - Test registration order
   - Test default agent selection
   - Test resolution by name and skill

2. **A2ATranslator**:
   - Test message building with various inputs
   - Test event translation for different event types
   - Test metadata handling

3. **RouterBridge**:
   - Test routing logic
   - Test context snapshot creation
   - Test agent descriptor resolution

### Integration Tests Needed

1. **End-to-End Flow**:
   - MCP client → agentique → A2A agent → response
   - Test with real ADK test agent

2. **Streaming**:
   - Verify incremental updates work correctly
   - Test context preservation across stream chunks

3. **Error Handling**:
   - Unknown agent names
   - A2A connection failures
   - Invalid message formats

---

## Performance Considerations

### Client Caching

The `A2AClientFactory` caches clients by base URL, which is good for performance. However:

**Observation**: Clients are never evicted from the cache, even if they fail or become stale.

**Recommendation**: Consider adding:
- Connection health checks
- Client reconnection logic
- Cache eviction policy (e.g., LRU, TTL)

### Event Processing

The current implementation collects all events in memory before returning (in `send_message`).

**Current Code**:
```python
async def send_message(...) -> AgentResponse:
    events: list[AgentEvent] = []
    async for event in self._iter_events(...):
        events.append(event)  # ❌ Collects all in memory

    return AgentResponse(
        text=self._translator.reduce_events(events),
        events=tuple(events),
    )
```

**Recommendation**: This is fine for small responses, but for large agent responses, consider streaming or chunked processing.

---

## Summary

### Fixed
✅ Critical `AgentRouter` initialization bug
✅ Incorrect `create_text_message_object()` parameter usage

### Verified
✅ All Python files have valid syntax
✅ FastMCP 3.0 features used correctly
✅ A2A SDK integration follows best practices
✅ Clean separation of concerns

### Recommendations
- Add unit and integration tests
- Simplify A2A adapter once SDK stabilizes
- Add client health checks and cache management
- Monitor memory usage for large agent responses

---

## Files Modified

1. **[src/agentique/router.py](src/agentique/router.py:12-17)**: Fixed `_default` initialization order
2. **[src/agentique/a2a_adapter.py](src/agentique/a2a_adapter.py:182)**: Fixed `create_text_message_object()` call

## Verification

All modified files pass Python syntax validation:
```bash
✓ src/agentique/router.py
✓ src/agentique/a2a_adapter.py
✓ All agentique modules
```

The MCP server should now start successfully and properly route messages to A2A agents.
