/**
 * ConversationSimulator - handles visual animations for the simulation observer UI.
 *
 * This class manages typing animations, thinking indicators, and reading effects
 * for the simulated conversation. All animations are visual-only (the actual
 * simulation runs server-side).
 */
class ConversationSimulator {
  constructor() {
    this.speed = 1.0;
    this.paused = false;
    this._pauseResolve = null;
  }

  /**
   * Animate typing in the visual input area, character by character.
   */
  async animateTyping(message, typingSpeedMs = 80) {
    const typingArea = document.getElementById('typingArea');
    const typingText = document.getElementById('typingText');

    typingArea.classList.add('active');
    typingText.textContent = '';

    for (let i = 0; i < message.length; i++) {
      if (this.paused) await this._waitForResume();
      typingText.textContent = message.substring(0, i + 1);
      const delay = (typingSpeedMs + (Math.random() * 40 - 20)) / this.speed;
      await this.sleep(Math.min(delay, 150));
    }

    // Brief pause after typing completes
    await this.sleep(300 / this.speed);
    typingArea.classList.remove('active');
    typingText.textContent = '';
  }

  /**
   * Show a thinking indicator for the simulated human.
   */
  showHumanThinking(personaName) {
    const messages = document.getElementById('messages');
    const bubble = document.createElement('div');
    bubble.className = 'thinking-bubble human';
    bubble.id = 'humanThinking';
    bubble.innerHTML = `
      <span>${personaName} is thinking</span>
      <div class="thinking-dots">
        <span>.</span><span>.</span><span>.</span>
      </div>
    `;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  /**
   * Show a thinking indicator for the AI agent.
   */
  showAIThinking() {
    const messages = document.getElementById('messages');
    const bubble = document.createElement('div');
    bubble.className = 'thinking-bubble agent';
    bubble.id = 'aiThinking';
    bubble.innerHTML = `
      <span>AI responding</span>
      <div class="thinking-dots">
        <span>.</span><span>.</span><span>.</span>
      </div>
    `;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  /**
   * Remove a thinking indicator.
   */
  hideThinking(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.remove();
  }

  /**
   * Add a completed human message bubble.
   */
  addHumanMessage(message, personaName) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message human';
    div.innerHTML = `<span class="sender">${this._escapeHtml(personaName)}</span>${this._escapeHtml(message)}`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  /**
   * Add an AI agent response bubble.
   */
  addAgentMessage(message) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message agent';
    div.innerHTML = `<span class="sender">AI Agent</span>${this._escapeHtml(message)}`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  /**
   * Add a system message.
   */
  addSystemMessage(text) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message system';
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  /**
   * Animate a reading effect on a message element.
   */
  async animateReading(element, durationMs = 1000) {
    if (!element) return;
    element.classList.add('being-read');
    await this.sleep(Math.min(durationMs, 2000) / this.speed);
    element.classList.remove('being-read');
  }

  /**
   * Set animation speed multiplier.
   */
  setSpeed(speed) {
    this.speed = speed;
  }

  /**
   * Pause animations.
   */
  pause() {
    this.paused = true;
  }

  /**
   * Resume animations.
   */
  resume() {
    this.paused = false;
    if (this._pauseResolve) {
      this._pauseResolve();
      this._pauseResolve = null;
    }
  }

  /**
   * Clear the chat area.
   */
  clearChat() {
    const messages = document.getElementById('messages');
    messages.innerHTML = '';
  }

  /**
   * Sleep for the given duration, respecting pause state.
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, Math.max(ms, 0)));
  }

  _waitForResume() {
    return new Promise(resolve => {
      this._pauseResolve = resolve;
    });
  }

  _escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Global instance
window.simulator = new ConversationSimulator();
