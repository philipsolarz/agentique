import { useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import ChatPanel from "./components/ChatPanel";
import CostDisplay from "./components/CostDisplay";
import RecursionTree from "./components/RecursionTree";
import type { StepInfo } from "./components/RecursionTree";
import ArtifactViewer from "./components/ArtifactViewer";
import type { Artifact } from "./components/ArtifactViewer";
import SessionSidebar from "./components/SessionSidebar";
import "./App.css";

interface SessionInfo {
  session_id: string;
  model: string;
  budget: number;
  name: string;
  created_at: string;
}

function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [cost, setCost] = useState(0);
  const [steps, setSteps] = useState<StepInfo[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  // Session creation form state
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [budget, setBudget] = useState("5.00");
  const [creating, setCreating] = useState(false);

  const activeBudget =
    sessions.find((s) => s.session_id === activeSessionId)?.budget ??
    (parseFloat(budget) || 5.0);

  const handleStepProgress = useCallback((step: StepInfo) => {
    setSteps((prev) => [...prev, step]);
  }, []);

  const handleArtifact = useCallback((artifact: Artifact) => {
    setArtifacts((prev) => [...prev, artifact]);
  }, []);

  function handleSelectSession(sessionId: string) {
    setActiveSessionId(sessionId);
    setCost(0);
    setSteps([]);
    setArtifacts([]);
  }

  function handleDeleteSession(sessionId: string) {
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
      setCost(0);
      setSteps([]);
      setArtifacts([]);
    }
  }

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
      setSteps([]);
      setArtifacts([]);
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
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
      />

      <div className="main-panel">
        <header className="top-bar">
          <h1>Agentique</h1>
          <CostDisplay cost={cost} budget={activeBudget} steps={steps} />
        </header>

        {activeSessionId ? (
          <>
            <RecursionTree steps={steps} />
            <ArtifactViewer artifacts={artifacts} />
            <ChatPanel
              sessionId={activeSessionId}
              onCostUpdate={setCost}
              onStepProgress={handleStepProgress}
              onArtifact={handleArtifact}
            />
          </>
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
