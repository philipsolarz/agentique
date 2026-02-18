/**
 * Visual Simulation Engine - Makes scenarios animate like a human using the UI
 *
 * Features:
 * - Character-by-character typing animation
 * - Button click animations
 * - Thinking indicators
 * - Realistic timing/pauses
 * - Playback speed controls
 */

class VisualSimulator {
  constructor() {
    this.isSimulating = false;
    this.isPaused = false;
    this.speed = 1.0; // 1.0 = normal, 2.0 = 2x speed, 0.5 = slow-mo
    this.queue = [];
    this.typingElement = null;
  }

  /**
   * Start simulating a scenario
   */
  async startScenario(scenarioName) {
    this.isSimulating = true;
    this.isPaused = false;
    this.updateSimulationUI(true);
  }

  /**
   * Stop simulation
   */
  stop() {
    this.isSimulating = false;
    this.isPaused = false;
    this.queue = [];
    this.updateSimulationUI(false);
    if (this.typingElement) {
      this.typingElement.remove();
      this.typingElement = null;
    }
  }

  /**
   * Pause/resume simulation
   */
  togglePause() {
    this.isPaused = !this.isPaused;
    this.updatePauseUI();
  }

  /**
   * Set simulation speed (0.5 = slow, 1.0 = normal, 2.0 = fast, 5.0 = very fast)
   */
  setSpeed(speed) {
    this.speed = Math.max(0.1, Math.min(10, speed));
    this.updateSpeedUI();
  }

  /**
   * Simulate typing a message character by character
   */
  async typeMessage(message) {
    const input = document.getElementById('messageInput');
    if (!input) return;

    // Clear existing input
    input.value = '';
    input.focus();

    // Add typing indicator in messages
    this.typingElement = this.addTypingIndicator();

    // Type each character with realistic delays
    for (let i = 0; i < message.length; i++) {
      if (!this.isSimulating) break;
      while (this.isPaused) {
        await this.sleep(100);
      }

      input.value += message[i];

      // Trigger input event for visual feedback
      input.dispatchEvent(new Event('input', { bubbles: true }));

      // Variable typing speed (humans don't type uniformly)
      const baseDelay = 80; // ms per character
      const variance = Math.random() * 40 - 20; // +/- 20ms
      const delay = (baseDelay + variance) / this.speed;

      await this.sleep(delay);
    }

    // Remove typing indicator
    if (this.typingElement) {
      this.typingElement.remove();
      this.typingElement = null;
    }

    // Small pause before "clicking" send
    await this.sleep(300 / this.speed);
  }

  /**
   * Simulate clicking the send button
   */
  async clickSend() {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;

    // Visual click effect
    sendBtn.classList.add('simulated-click');
    await this.sleep(150 / this.speed);
    sendBtn.classList.remove('simulated-click');

    // Actually click it
    sendBtn.click();
  }

  /**
   * Simulate selecting a scenario from dropdown
   */
  async selectScenario(scenarioName) {
    const select = document.getElementById('scenarioSelect');
    if (!select) return;

    // Highlight the dropdown
    select.focus();
    select.classList.add('simulated-select');
    await this.sleep(500 / this.speed);

    // Select the option
    select.value = scenarioName;
    select.dispatchEvent(new Event('change', { bubbles: true }));

    await this.sleep(300 / this.speed);
    select.classList.remove('simulated-select');
  }

  /**
   * Simulate clicking run scenario button
   */
  async clickRunScenario() {
    const btn = document.getElementById('runScenarioBtn');
    if (!btn) return;

    btn.classList.add('simulated-click');
    await this.sleep(150 / this.speed);
    btn.classList.remove('simulated-click');

    btn.click();
  }

  /**
   * Show "thinking" animation while waiting for response
   */
  showThinking() {
    const indicator = document.createElement('div');
    indicator.className = 'thinking-indicator';
    indicator.innerHTML = `
      <span class="thinking-avatar">🤖</span>
      <div class="thinking-dots">
        <span>.</span><span>.</span><span>.</span>
      </div>
    `;

    const messagesEl = document.getElementById('messages');
    if (messagesEl) {
      messagesEl.appendChild(indicator);
      this.scrollMessages();
    }

    return indicator;
  }

  /**
   * Remove thinking indicator
   */
  hideThinking(indicator) {
    if (indicator && indicator.parentNode) {
      indicator.remove();
    }
  }

  /**
   * Add a typing indicator (shows user is "typing")
   */
  addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
      <span class="typing-avatar">👤</span>
      <div class="typing-text">typing...</div>
    `;

    const messagesEl = document.getElementById('messages');
    if (messagesEl) {
      messagesEl.appendChild(indicator);
      this.scrollMessages();
    }

    return indicator;
  }

  /**
   * Simulate reading a response (highlight it briefly)
   */
  async readResponse(messageElement) {
    if (!messageElement) return;

    messageElement.classList.add('being-read');
    // Simulate reading time based on length
    const readTime = Math.min(3000, messageElement.textContent.length * 20);
    await this.sleep(readTime / this.speed);
    messageElement.classList.remove('being-read');
  }

  /**
   * Complete simulation of sending a message
   */
  async simulateSendMessage(message) {
    if (!this.isSimulating) return;

    // Type the message
    await this.typeMessage(message);

    // Click send
    await this.clickSend();

    // Show thinking while waiting for response
    const thinking = this.showThinking();

    // Wait for response (this will be removed when actual response arrives)
    // The response handling is done by the normal message handler

    // Small pause after response
    await this.sleep(500 / this.speed);

    this.hideThinking(thinking);
  }

  /**
   * Update UI to show simulation is active
   */
  updateSimulationUI(active) {
    const controls = document.getElementById('simulationControls');
    if (controls) {
      controls.style.display = active ? 'flex' : 'none';
    }

    // Disable manual controls during simulation
    const manualControls = [
      'messageInput',
      'sendBtn',
      'scenarioSelect',
      'runScenarioBtn',
      'runAllBtn'
    ];

    for (const id of manualControls) {
      const el = document.getElementById(id);
      if (el) {
        el.disabled = active;
      }
    }
  }

  /**
   * Update pause button UI
   */
  updatePauseUI() {
    const pauseBtn = document.getElementById('pauseSimulationBtn');
    if (pauseBtn) {
      pauseBtn.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
    }
  }

  /**
   * Update speed display
   */
  updateSpeedUI() {
    const speedDisplay = document.getElementById('speedDisplay');
    if (speedDisplay) {
      speedDisplay.textContent = `${this.speed.toFixed(1)}x`;
    }
  }

  /**
   * Helper to scroll messages panel
   */
  scrollMessages() {
    const messagesEl = document.getElementById('messages');
    if (messagesEl) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  /**
   * Helper sleep function that respects pause state
   */
  async sleep(ms) {
    const start = Date.now();
    while (Date.now() - start < ms) {
      if (!this.isSimulating) break;
      while (this.isPaused && this.isSimulating) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      await new Promise(resolve => setTimeout(resolve, Math.min(50, ms)));
    }
  }
}

// Create global simulator instance
window.visualSimulator = new VisualSimulator();
