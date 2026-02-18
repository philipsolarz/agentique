/**
 * Agentique Test Client — WebSocket client, message rendering, scenario controls.
 */

(function () {
  "use strict";

  // DOM elements
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const mcpUrlEl = document.getElementById("mcpUrl");
  const messagesEl = document.getElementById("messages");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const scenarioSelect = document.getElementById("scenarioSelect");
  const runScenarioBtn = document.getElementById("runScenarioBtn");
  const runAllBtn = document.getElementById("runAllBtn");
  const eventLog = document.getElementById("eventLog");
  const clearEventsBtn = document.getElementById("clearEventsBtn");
  const elicitModal = document.getElementById("elicitModal");
  const elicitMessage = document.getElementById("elicitMessage");
  const elicitInput = document.getElementById("elicitInput");
  const elicitSubmitBtn = document.getElementById("elicitSubmitBtn");
  const elicitCancelBtn = document.getElementById("elicitCancelBtn");
  const startAutonomousBtn = document.getElementById("startAutonomousBtn");
  const autonomousStatus = document.getElementById("autonomousStatus");
  const autonomousStatusMessage = document.getElementById("autonomousStatusMessage");

  let ws = null;
  let connected = false;
  let currentThinkingIndicator = null;
  let simulationMode = true;

  // --- WebSocket Connection ---

  function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}/ws`;
    ws = new WebSocket(url);

    ws.onopen = function () {
      connected = true;
      setStatus(true);
      enableControls(true);
      addSystemMessage("Connected to test harness.");
      // Request available scenarios
      wsSend({ type: "list_scenarios" });
    };

    ws.onclose = function () {
      connected = false;
      setStatus(false);
      enableControls(false);
      addSystemMessage("Disconnected from test harness.");
      // Reconnect after delay
      setTimeout(connect, 3000);
    };

    ws.onerror = function () {
      // onclose will fire after this
    };

    ws.onmessage = function (evt) {
      try {
        const msg = JSON.parse(evt.data);
        handleMessage(msg);
      } catch (e) {
        console.error("Failed to parse message:", e);
      }
    };
  }

  function wsSend(msg) {
    if (ws && connected) {
      ws.send(JSON.stringify(msg));
    }
  }

  // --- Message Handling ---

  function handleMessage(msg) {
    switch (msg.type) {
      case "connected":
        mcpUrlEl.textContent = msg.data.mcp_url || "";
        addSystemMessage("MCP client connected to: " + (msg.data.mcp_url || "unknown"));
        break;

      case "tools_listed":
        addSystemMessage("Tools: " + (msg.data.tools || []).join(", "));
        break;

      case "tool_call":
        addToolCall(msg.data);
        break;

      case "tool_result":
        addToolResult(msg.data);
        break;

      case "log_message":
        // Show in event log only
        break;

      case "progress":
        showProgress(msg.data);
        break;

      case "elicitation_request":
        showElicitation(msg.data.message);
        break;

      case "scenario_list":
        populateScenarios(msg.data.scenarios || []);
        break;

      case "scenario_result":
        showScenarioResult(msg.data);
        if (window.visualSimulator && window.visualSimulator.isSimulating) {
          window.visualSimulator.stop();
        }
        break;

      case "simulation_event":
        if (window.handleSimulationEvent) {
          window.handleSimulationEvent(msg.data);
        }
        break;

      case "autonomous_agent_status":
        handleAutonomousStatus(msg.data);
        break;

      case "autonomous_agent_action":
        showAutonomousAction(msg.data);
        break;

      case "autonomous_agent_insight":
        showAutonomousInsight(msg.data);
        break;

      case "autonomous_agent_report":
        showAutonomousReport(msg.data);
        break;

      case "error":
        addErrorMessage(msg.data.error || "Unknown error");
        break;

      default:
        break;
    }

    // Always log the event
    addEventEntry(msg);
  }

  // --- UI Updates ---

  function setStatus(isConnected) {
    statusDot.classList.toggle("connected", isConnected);
    statusText.textContent = isConnected ? "Connected" : "Disconnected";
  }

  function enableControls(enabled) {
    messageInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    scenarioSelect.disabled = !enabled;
    runScenarioBtn.disabled = !enabled;
    runAllBtn.disabled = !enabled;
    if (startAutonomousBtn) {
      startAutonomousBtn.disabled = !enabled;
    }
  }

  function scrollMessages() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message user";
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollMessages();
  }

  function addAgentMessage(text) {
    const el = document.createElement("div");
    el.className = "message agent";
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollMessages();
  }

  function addErrorMessage(text) {
    const el = document.createElement("div");
    el.className = "message error";
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollMessages();
  }

  function addSystemMessage(text) {
    const el = document.createElement("div");
    el.className = "message system";
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollMessages();
  }

  function addToolCall(data) {
    const card = document.createElement("div");
    card.className = "tool-call-card";
    card.id = "tool-call-" + Date.now();

    const nameEl = document.createElement("div");
    nameEl.className = "tool-name";
    nameEl.textContent = data.tool || "unknown";
    card.appendChild(nameEl);

    if (data.arguments && Object.keys(data.arguments).length > 0) {
      const argsEl = document.createElement("div");
      argsEl.className = "tool-args";
      argsEl.textContent = JSON.stringify(data.arguments, null, 2);
      card.appendChild(argsEl);
    }

    card._toolName = data.tool;
    messagesEl.appendChild(card);
    scrollMessages();
  }

  function addToolResult(data) {
    // Find the most recent tool call card for this tool
    const cards = messagesEl.querySelectorAll(".tool-call-card");
    let targetCard = null;
    for (let i = cards.length - 1; i >= 0; i--) {
      if (cards[i]._toolName === data.tool && !cards[i]._hasResult) {
        targetCard = cards[i];
        break;
      }
    }

    if (targetCard) {
      const resultEl = document.createElement("div");
      resultEl.className = "tool-result" + (data.is_error ? " error" : "");
      resultEl.textContent = data.text || "(empty response)";
      targetCard.appendChild(resultEl);
      targetCard._hasResult = true;
    } else {
      // No matching card — show as agent message
      if (data.is_error) {
        addErrorMessage(data.text || "Tool error");
      } else {
        addAgentMessage(data.text || "(empty response)");
      }
    }
    scrollMessages();
  }

  function showProgress(data) {
    // Find or create progress bar
    let bar = messagesEl.querySelector(".progress-bar:last-child");
    if (!bar) {
      bar = document.createElement("div");
      bar.className = "progress-bar";
      bar.innerHTML =
        '<div class="progress-bar-track"><div class="progress-bar-fill"></div></div>' +
        '<div class="progress-bar-label"></div>';
      messagesEl.appendChild(bar);
    }

    const fill = bar.querySelector(".progress-bar-fill");
    const label = bar.querySelector(".progress-bar-label");
    const pct = data.total ? Math.round((data.progress / data.total) * 100) : 0;
    fill.style.width = pct + "%";
    label.textContent = (data.message || "Progress") + " " + pct + "%";
    scrollMessages();
  }

  // --- Elicitation ---

  function showElicitation(message) {
    elicitMessage.textContent = message || "The agent is requesting input.";
    elicitInput.value = "";
    elicitModal.classList.add("active");
    elicitInput.focus();
  }

  function hideElicitation() {
    elicitModal.classList.remove("active");
  }

  elicitSubmitBtn.addEventListener("click", function () {
    const value = elicitInput.value;
    wsSend({
      type: "elicitation_response",
      data: { response: { action: "submit", value: value } },
    });
    hideElicitation();
  });

  elicitCancelBtn.addEventListener("click", function () {
    wsSend({
      type: "elicitation_response",
      data: { response: { action: "cancel" } },
    });
    hideElicitation();
  });

  elicitInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      elicitSubmitBtn.click();
    }
  });

  // --- Event Log ---

  function addEventEntry(msg) {
    const entry = document.createElement("div");
    entry.className = "event-entry";

    const timeEl = document.createElement("span");
    timeEl.className = "event-time";
    const now = new Date();
    timeEl.textContent =
      now.getHours().toString().padStart(2, "0") +
      ":" +
      now.getMinutes().toString().padStart(2, "0") +
      ":" +
      now.getSeconds().toString().padStart(2, "0") +
      "." +
      now.getMilliseconds().toString().padStart(3, "0");

    const typeEl = document.createElement("span");
    typeEl.className = "event-type " + (msg.type || "");
    typeEl.textContent = msg.type || "unknown";

    const dataEl = document.createElement("span");
    dataEl.className = "event-data";
    dataEl.textContent = summarizeEventData(msg);

    entry.appendChild(timeEl);
    entry.appendChild(typeEl);
    entry.appendChild(dataEl);
    eventLog.appendChild(entry);
    eventLog.scrollTop = eventLog.scrollHeight;
  }

  function summarizeEventData(msg) {
    const d = msg.data || {};
    switch (msg.type) {
      case "tool_call":
        return d.tool + "(" + JSON.stringify(d.arguments || {}) + ")";
      case "tool_result":
        return (d.is_error ? "ERROR: " : "") + (d.text || "").slice(0, 100);
      case "log_message":
        return "[" + (d.level || "info") + "] " + (typeof d.data === "string" ? d.data : JSON.stringify(d.data)).slice(0, 100);
      case "progress":
        return (d.progress || 0) + "/" + (d.total || "?") + " " + (d.message || "");
      case "error":
        return d.error || "";
      case "scenario_result":
        return d.scenario_name + ": " + (d.passed ? "PASS" : "FAIL");
      case "elicitation_request":
        return d.message || "";
      default:
        return JSON.stringify(d).slice(0, 120);
    }
  }

  clearEventsBtn.addEventListener("click", function () {
    eventLog.innerHTML = "";
  });

  // --- Scenarios ---

  function populateScenarios(scenarios) {
    scenarioSelect.innerHTML = '<option value="">-- select --</option>';
    scenarios.forEach(function (s) {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      scenarioSelect.appendChild(opt);
    });
  }

  function showScenarioResult(data) {
    const container = document.createElement("div");
    container.className = "scenario-results";

    const title = document.createElement("h3");
    title.textContent = "Scenario: " + (data.scenario_name || "unknown");
    container.appendChild(title);

    (data.step_results || []).forEach(function (step) {
      const row = document.createElement("div");
      row.className = "step-result";

      const status = document.createElement("span");
      status.className = "step-status " + (step.passed ? "pass" : "fail");
      status.textContent = step.passed ? "PASS" : "FAIL";

      const name = document.createElement("span");
      name.className = "step-name";
      name.textContent = step.step_name;

      const time = document.createElement("span");
      time.className = "step-time";
      time.textContent = Math.round(step.response_time_ms) + "ms";

      row.appendChild(status);
      row.appendChild(name);
      row.appendChild(time);
      container.appendChild(row);
    });

    const summary = document.createElement("div");
    summary.className = "summary " + (data.passed ? "pass" : "fail");
    summary.textContent =
      (data.passed ? "ALL PASSED" : "FAILED") +
      " (" +
      Math.round(data.total_time_ms) +
      "ms)";
    container.appendChild(summary);

    messagesEl.appendChild(container);
    scrollMessages();
  }

  // --- Send Message ---

  function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;

    addUserMessage(text);
    wsSend({ type: "send_message", data: { message: text } });
    messageInput.value = "";
  }

  sendBtn.addEventListener("click", sendMessage);
  messageInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  runScenarioBtn.addEventListener("click", async function () {
    const name = scenarioSelect.value;
    if (!name) return;

    if (simulationMode && window.visualSimulator) {
      // Visual simulation mode
      await window.visualSimulator.startScenario(name);
      addSystemMessage("🎬 Starting visual simulation: " + name);
      wsSend({ type: "run_scenario", data: { name: name, simulate: true } });
    } else {
      // Headless mode
      addSystemMessage("Running scenario: " + name);
      wsSend({ type: "run_scenario", data: { name: name, simulate: false } });
    }
  });

  runAllBtn.addEventListener("click", function () {
    addSystemMessage("Running all scenarios...");
    wsSend({ type: "run_all_scenarios", data: {} });
  });

  // --- Autonomous Agent ---

  if (startAutonomousBtn) {
    startAutonomousBtn.addEventListener("click", function () {
      addSystemMessage("🤖 Starting AI-powered autonomous testing...");
      wsSend({
        type: "start_autonomous_agent",
        data: {
          objective: {
            goal: "Comprehensively test all MCP server capabilities",
            focus_areas: [
              "tool calling",
              "agent routing",
              "error handling",
              "streaming responses"
            ]
          },
          max_actions: 15
        }
      });
    });
  }

  function handleAutonomousStatus(data) {
    if (!autonomousStatus || !autonomousStatusMessage) return;

    const status = data.status || "";
    const message = data.message || "";

    autonomousStatusMessage.textContent = message;

    switch (status) {
      case "starting":
      case "exploring":
        autonomousStatus.style.display = "flex";
        if (startAutonomousBtn) startAutonomousBtn.disabled = true;
        break;

      case "completed":
      case "failed":
        autonomousStatus.style.display = "none";
        if (startAutonomousBtn) startAutonomousBtn.disabled = false;
        addSystemMessage("🤖 " + message);
        break;
    }
  }

  function showAutonomousAction(data) {
    const card = document.createElement("div");
    card.className = "ai-action-card";

    const icon = data.action_type === "send_message" ? "💬" : "🔧";

    card.innerHTML = `
      <div class="action-header">
        <span class="action-type">${data.action_type}</span>
        <span class="action-icon">${icon}</span>
      </div>
      <div class="action-reasoning">${data.reasoning || ""}</div>
      <div class="action-params">${JSON.stringify(data.parameters).slice(0, 100)}...</div>
    `;

    messagesEl.appendChild(card);
    scrollMessages();
  }

  function showAutonomousInsight(data) {
    const card = document.createElement("div");
    card.className = "ai-insight-card severity-" + (data.severity || "info");

    const evidenceHTML = (data.evidence || []).map(e =>
      `<div>• ${e}</div>`
    ).join("");

    card.innerHTML = `
      <div class="insight-header">
        <span class="insight-category">${data.category || "insight"}</span>
        <span class="insight-severity">${data.severity || "info"}</span>
      </div>
      <div class="insight-description">${data.description || ""}</div>
      ${evidenceHTML ? `<div class="insight-evidence">${evidenceHTML}</div>` : ""}
    `;

    messagesEl.appendChild(card);
    scrollMessages();
  }

  function showAutonomousReport(data) {
    const card = document.createElement("div");
    card.className = "ai-report-card";

    const successRate = ((data.success_rate || 0) * 100).toFixed(1);
    const coverage = data.coverage || {};
    const coveragePercent = coverage.total_tools > 0
      ? ((coverage.tools_tested / coverage.total_tools) * 100).toFixed(1)
      : "0";

    const insightsHTML = (data.insights || []).slice(0, 3).map(insight =>
      `<div style="margin: 4px 0; color: var(--text-muted);">
        <strong style="color: var(--accent-yellow);">[${insight.category}]</strong> ${insight.description}
      </div>`
    ).join("");

    card.innerHTML = `
      <div class="report-title">🎉 Autonomous Testing Complete</div>
      <div class="report-stats">
        <div class="stat">
          <span class="stat-label">Actions Executed</span>
          <span class="stat-value">${data.actions_executed || 0}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Success Rate</span>
          <span class="stat-value">${successRate}%</span>
        </div>
        <div class="stat">
          <span class="stat-label">Insights Found</span>
          <span class="stat-value">${data.insights_discovered || 0}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Tool Coverage</span>
          <span class="stat-value">${coveragePercent}%</span>
        </div>
      </div>
      ${insightsHTML ? `
        <div class="report-insights">
          <div class="insights-title">Top Insights:</div>
          ${insightsHTML}
        </div>
      ` : ""}
    `;

    messagesEl.appendChild(card);
    scrollMessages();
  }

  // --- Simulation Controls ---

  const simulationModeCheckbox = document.getElementById("simulationMode");
  const pauseSimBtn = document.getElementById("pauseSimulationBtn");
  const stopSimBtn = document.getElementById("stopSimulationBtn");
  const speedSlider = document.getElementById("speedSlider");

  if (simulationModeCheckbox) {
    simulationModeCheckbox.addEventListener("change", function () {
      simulationMode = this.checked;
    });
  }

  if (pauseSimBtn && window.visualSimulator) {
    pauseSimBtn.addEventListener("click", function () {
      window.visualSimulator.togglePause();
    });
  }

  if (stopSimBtn && window.visualSimulator) {
    stopSimBtn.addEventListener("click", function () {
      window.visualSimulator.stop();
      addSystemMessage("Simulation stopped");
    });
  }

  if (speedSlider && window.visualSimulator) {
    speedSlider.addEventListener("input", function () {
      window.visualSimulator.setSpeed(parseFloat(this.value));
    });
  }

  // --- Handle Simulation Events ---

  function handleSimulationEvent(event) {
    if (!window.visualSimulator || !window.visualSimulator.isSimulating) return;

    const sim = window.visualSimulator;

    switch (event.action) {
      case "type_message":
        sim.simulateSendMessage(event.message);
        break;

      case "show_thinking":
        currentThinkingIndicator = sim.showThinking();
        break;

      case "hide_thinking":
        if (currentThinkingIndicator) {
          sim.hideThinking(currentThinkingIndicator);
          currentThinkingIndicator = null;
        }
        break;

      case "read_response":
        const lastMessage = messagesEl.querySelector(".message.agent:last-child");
        if (lastMessage) {
          sim.readResponse(lastMessage);
        }
        break;
    }
  }

  // Expose for WebSocket message handler
  window.handleSimulationEvent = handleSimulationEvent;

  // --- Init ---
  connect();
})();
