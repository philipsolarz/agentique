/**
 * Simulation Observer UI - WebSocket event handler and UI updates.
 *
 * Connects to the Simulation UI server via WebSocket and receives all
 * simulation events in real-time. No user input - pure observation.
 */

let ws = null;
let connected = false;
let currentSimulationId = null;
let currentPersonaName = 'Alex';
let turnCount = 0;
let insightCount = 0;

// --- WebSocket Connection ---

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus(true);
  };

  ws.onclose = () => {
    setStatus(false);
    // Auto-reconnect after 3s
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    setStatus(false);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch (e) {
      console.error('Failed to parse message:', e);
    }
  };
}

function setStatus(isConnected) {
  connected = isConnected;
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.className = isConnected ? 'status-dot connected' : 'status-dot';
  text.textContent = isConnected ? 'Connected' : 'Disconnected';
}

// --- Message Router ---

function handleMessage(msg) {
  const { type, data, simulation_id } = msg;

  // Track simulation ID
  if (simulation_id && !currentSimulationId) {
    currentSimulationId = simulation_id;
  }

  // Add to event log
  addEventEntry(msg);

  // Route to handler
  switch (type) {
    case 'connected':
      document.getElementById('mcpUrl').textContent = data.mcp_url || '';
      break;

    case 'simulation_started':
      handleSimulationStarted(data);
      break;

    case 'thinking':
      handleThinking(data);
      break;

    case 'typing':
      handleTyping(data);
      break;

    case 'message_sent':
      handleMessageSent(data);
      break;

    case 'waiting':
      handleWaiting(data);
      break;

    case 'response_received':
      handleResponseReceived(data);
      break;

    case 'reading':
      handleReading(data);
      break;

    case 'turn_complete':
      handleTurnComplete(data);
      break;

    case 'insight':
      handleInsight(data);
      break;

    case 'summary':
      handleSummary(data);
      break;

    case 'simulation_completed':
      handleSimulationCompleted(data);
      break;

    case 'simulation_failed':
      handleSimulationFailed(data);
      break;

    case 'simulation_stopped':
      handleSimulationStopped(data);
      break;

    // MCP protocol events (shown in event log only)
    case 'tool_call':
    case 'tool_result':
    case 'log_message':
    case 'progress':
    case 'error':
      // Already added to event log above
      break;
  }
}

// --- Simulation Event Handlers ---

function handleSimulationStarted(data) {
  const persona = data.persona || {};
  const objective = data.objective || {};

  currentPersonaName = persona.name || 'Alex';
  turnCount = 0;
  insightCount = 0;

  // Clear chat
  const messages = document.getElementById('messages');
  messages.innerHTML = '';

  // Show simulation banner
  const banner = document.getElementById('simulationBanner');
  banner.classList.add('active');
  document.getElementById('simPersona').textContent = `${currentPersonaName} (${persona.role || 'user'})`;
  document.getElementById('simObjective').textContent = objective.goal || '';
  document.getElementById('simTurnCount').textContent = '0';
  document.getElementById('simState').textContent = 'running';
  document.getElementById('simInsights').textContent = '0';

  // Show simulation info in event panel
  const info = document.getElementById('simulationInfo');
  info.classList.add('active');
  document.getElementById('infoPersona').textContent = `${currentPersonaName} (${persona.role || 'user'})`;
  document.getElementById('infoObjective').textContent = objective.goal || '';
  updateInfoState('running');
  document.getElementById('infoTurns').textContent = '0';

  // System message
  window.simulator.addSystemMessage(
    `Simulation started: ${currentPersonaName} will explore "${objective.goal || 'AI capabilities'}"`
  );
}

function handleThinking(data) {
  const name = data.persona || currentPersonaName;
  window.simulator.showHumanThinking(name);
}

function handleTyping(data) {
  window.simulator.hideThinking('humanThinking');
  const message = data.message || '';
  const speed = data.typing_speed_ms || 80;
  window.simulator.animateTyping(message, speed);
}

function handleMessageSent(data) {
  window.simulator.hideThinking('humanThinking');
  const message = data.message || '';
  window.simulator.addHumanMessage(message, currentPersonaName);
}

function handleWaiting(data) {
  window.simulator.showAIThinking();
}

function handleResponseReceived(data) {
  window.simulator.hideThinking('aiThinking');
  const message = data.message || '';
  const isError = data.is_error || false;

  if (isError) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message error';
    div.textContent = message;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  } else {
    window.lastAgentMessage = window.simulator.addAgentMessage(message);
  }
}

function handleReading(data) {
  if (window.lastAgentMessage) {
    const duration = data.duration_ms || 1000;
    window.simulator.animateReading(window.lastAgentMessage, duration);
  }
}

function handleTurnComplete(data) {
  turnCount = data.turn || turnCount + 1;
  document.getElementById('simTurnCount').textContent = turnCount;
  document.getElementById('infoTurns').textContent = turnCount;
}

function handleInsight(data) {
  insightCount++;
  document.getElementById('simInsights').textContent = insightCount;

  // Add insight card to event log
  const eventLog = document.getElementById('eventLog');
  const card = document.createElement('div');
  card.className = `insight-card severity-${data.severity || 'info'}`;
  card.innerHTML = `
    <div class="insight-header">
      <span class="insight-category">${escapeHtml(data.category || 'insight')}</span>
      <span class="insight-severity">${escapeHtml(data.severity || 'info')}</span>
    </div>
    <div class="insight-description">${escapeHtml(data.description || '')}</div>
  `;
  eventLog.appendChild(card);
  eventLog.scrollTop = eventLog.scrollHeight;
}

function handleSummary(data) {
  const summary = data.summary || '';
  window.simulator.addSystemMessage(`Summary: ${summary}`);
}

function handleSimulationCompleted(data) {
  updateSimState('completed');

  // Show summary card in chat
  const messages = document.getElementById('messages');
  const card = document.createElement('div');
  card.className = 'summary-card';
  card.innerHTML = `
    <div class="summary-title">Simulation Complete</div>
    <div class="summary-stats">
      <div class="stat">
        <span class="stat-label">Turns</span>
        <span class="stat-value">${data.turns || turnCount}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Insights</span>
        <span class="stat-value">${data.insights_count || insightCount}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Duration</span>
        <span class="stat-value">${formatDuration(data.total_time_ms)}</span>
      </div>
    </div>
    <div class="summary-text">${escapeHtml(data.summary || 'Simulation completed.')}</div>
  `;
  messages.appendChild(card);
  messages.scrollTop = messages.scrollHeight;

  currentSimulationId = null;
}

function handleSimulationFailed(data) {
  updateSimState('failed');
  window.simulator.hideThinking('humanThinking');
  window.simulator.hideThinking('aiThinking');

  window.simulator.addSystemMessage(`Simulation failed: ${data.error || 'Unknown error'}`);
  currentSimulationId = null;
}

function handleSimulationStopped(data) {
  updateSimState('stopped');
  window.simulator.hideThinking('humanThinking');
  window.simulator.hideThinking('aiThinking');

  window.simulator.addSystemMessage(`Simulation stopped: ${data.reason || 'User requested'}`);
  currentSimulationId = null;
}

// --- UI Helpers ---

function updateSimState(state) {
  document.getElementById('simState').textContent = state;
  updateInfoState(state);
}

function updateInfoState(state) {
  const el = document.getElementById('infoState');
  el.textContent = state;
  el.className = `info-value state-${state}`;
}

function addEventEntry(msg) {
  const eventLog = document.getElementById('eventLog');
  const entry = document.createElement('div');
  entry.className = 'event-entry';

  const time = new Date(msg.timestamp * 1000).toLocaleTimeString();
  const type = msg.type || 'unknown';
  let dataStr = '';

  if (msg.data) {
    if (msg.data.message) {
      dataStr = msg.data.message.substring(0, 80);
    } else if (msg.data.error) {
      dataStr = msg.data.error.substring(0, 80);
    } else if (msg.data.tool) {
      dataStr = msg.data.tool;
    } else if (msg.data.summary) {
      dataStr = msg.data.summary.substring(0, 80);
    } else if (msg.data.persona) {
      dataStr = typeof msg.data.persona === 'string' ? msg.data.persona : (msg.data.persona.name || '');
    }
  }

  entry.innerHTML = `
    <span class="event-time">${time}</span>
    <span class="event-type ${type}">${type}</span>
    <span class="event-data">${escapeHtml(dataStr)}</span>
  `;
  eventLog.appendChild(entry);
  eventLog.scrollTop = eventLog.scrollHeight;
}

function clearEventLog() {
  document.getElementById('eventLog').innerHTML = '';
}

function updateSpeed(value) {
  window.simulator.setSpeed(parseFloat(value));
  document.getElementById('speedDisplay').textContent = value + 'x';
}

function formatDuration(ms) {
  if (!ms) return '-';
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes}m ${secs}s`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// --- Initialize ---
connect();
