import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import ChatPanel from "./components/ChatPanel";
import CostDisplay from "./components/CostDisplay";
import SessionSidebar from "./components/SessionSidebar";
import "./App.css";

interface SessionInfo {
  session_id: string;
  model: string;
  budget: number;
}

function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [cost, setCost] = useState(0);

  // Session creation form state
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [budget, setBudget] = useState("5.00");
  const [creating, setCreating] = useState(false);

  async function handleCreateSession() {
    if (!apiKey.trim()) return;
    setCreating(true);
    try {
      const info = await invoke<SessionInfo>("create_session", {
        apiKey: apiKey.trim(),
        model,
        budget: parseFloat(budget) || 5.0,
      });
      setSessions((prev) => [...prev, info]);
      setActiveSessionId(info.session_id);
      setCost(0);
    } catch (e) {
      console.error("Failed to create session:", e);
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="app">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={setActiveSessionId}
      />

      <div className="main-panel">
        <header className="top-bar">
          <h1>Agentique</h1>
          <CostDisplay cost={cost} />
        </header>

        {activeSessionId ? (
          <ChatPanel
            sessionId={activeSessionId}
            onCostUpdate={setCost}
          />
        ) : (
          <div className="setup-form">
            <h2>New Session</h2>
            <label>
              API Key
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
              />
            </label>
            <label>
              Model
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </label>
            <label>
              Budget (USD)
              <input
                type="text"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
              />
            </label>
            <button onClick={handleCreateSession} disabled={creating}>
              {creating ? "Creating..." : "Start Session"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
