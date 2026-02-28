import { useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import ChatPanel from "./components/ChatPanel";
import CostDisplay from "./components/CostDisplay";
import RecursionTree from "./components/RecursionTree";
import type { StepInfo } from "./components/RecursionTree";
import ArtifactViewer from "./components/ArtifactViewer";
import type { Artifact } from "./components/ArtifactViewer";
import SessionSidebar from "./components/SessionSidebar";
import Settings from "./components/Settings";
import SystemPromptEditor from "./components/SystemPromptEditor";
import ModelSelector from "./components/ModelSelector";
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
  const [baseUrl, setBaseUrl] = useState("");
  const [creating, setCreating] = useState(false);

  const [exporting, setExporting] = useState(false);
  const [currentModel, setCurrentModel] = useState(model);
  const [isSending, setIsSending] = useState(false);

  const activeBudget =
    sessions.find((s) => s.session_id === activeSessionId)?.budget ??
    (parseFloat(budget) || 5.0);

  const handleStepProgress = useCallback((step: StepInfo) => {
    setSteps((prev) => [...prev, step]);
  }, []);

  const handleArtifact = useCallback((artifact: Artifact) => {
    setArtifacts((prev) => [...prev, artifact]);
  }, []);

  async function handleExport(format: "markdown" | "json") {
    if (!activeSessionId || exporting) return;
    setExporting(true);
    try {
      const content = await invoke<string>("export_conversation", {
        sessionId: activeSessionId,
        format,
      });
      const ext = format === "json" ? "json" : "md";
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `agentique-${activeSessionId.slice(0, 8)}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    } finally {
      setExporting(false);
    }
  }

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
        baseUrl: baseUrl.trim() || null,
      });
      setSessions((prev) => [...prev, info]);
      setActiveSessionId(info.session_id);
      setCurrentModel(info.model);
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
        onResume={(info) => {
          setSessions((prev) => [...prev, info]);
          setActiveSessionId(info.session_id);
          setCost(0);
          setSteps([]);
          setArtifacts([]);
        }}
        onFork={(info) => {
          setSessions((prev) => [...prev, info]);
          setActiveSessionId(info.session_id);
          setCost(0);
          setSteps([]);
          setArtifacts([]);
        }}
        apiKey={apiKey}
        model={model}
        budget={parseFloat(budget) || 5.0}
      />

      <div className="main-panel">
        <header className="top-bar">
          <h1>Agentique</h1>
          <div className="header-actions">
            {activeSessionId && (
              <>
                <ModelSelector
                  sessionId={activeSessionId}
                  currentModel={currentModel}
                  disabled={isSending}
                  onModelSwitched={setCurrentModel}
                />
                <div className="export-buttons">
                  <button className="btn-export" onClick={() => handleExport("markdown")} disabled={exporting}>
                    Export MD
                  </button>
                  <button className="btn-export" onClick={() => handleExport("json")} disabled={exporting}>
                    Export JSON
                  </button>
                </div>
              </>
            )}
            <CostDisplay cost={cost} budget={activeBudget} steps={steps} />
          </div>
        </header>

        {activeSessionId ? (
          <>
            <SystemPromptEditor sessionId={activeSessionId} />
            <RecursionTree steps={steps} />
            <ArtifactViewer artifacts={artifacts} />
            <ChatPanel
              sessionId={activeSessionId}
              onCostUpdate={setCost}
              onStepProgress={handleStepProgress}
              onArtifact={handleArtifact}
              onSendingChange={setIsSending}
              onModelSwitched={setCurrentModel}
            />
          </>
        ) : (
          <div className="setup-form">
            <h2>New Session</h2>
            <Settings
              onSettingsLoaded={(s) => {
                if (s.api_key) setApiKey(s.api_key);
                if (s.model) setModel(s.model);
                if (s.budget) setBudget(String(s.budget));
                if (s.base_url) setBaseUrl(s.base_url);
              }}
              apiKey={apiKey}
              model={model}
              budget={budget}
              baseUrl={baseUrl}
              onApiKeyChange={setApiKey}
              onModelChange={setModel}
              onBudgetChange={setBudget}
              onBaseUrlChange={setBaseUrl}
            />
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
