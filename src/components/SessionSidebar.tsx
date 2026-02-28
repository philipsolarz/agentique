import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

interface SessionInfo {
  session_id: string;
  model: string;
  budget: number;
  name: string;
  created_at: string;
}

interface PersistedSessionInfo {
  session_id: string;
  name: string;
  created_at: string;
}

interface Props {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onResume: (session: SessionInfo) => void;
  onFork: (session: SessionInfo) => void;
  apiKey: string;
  model: string;
  budget: number;
}

function formatTimestamp(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const isToday =
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear();
  if (isToday) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function SessionSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onDelete,
  onResume,
  onFork,
  apiKey,
  model,
  budget,
}: Props) {
  const [persisted, setPersisted] = useState<PersistedSessionInfo[]>([]);
  const [showPersisted, setShowPersisted] = useState(false);

  useEffect(() => {
    loadPersisted();
  }, [sessions]);

  async function loadPersisted() {
    try {
      const list = await invoke<PersistedSessionInfo[]>("list_persisted_sessions");
      // Filter out sessions that are already active
      const activeIds = new Set(sessions.map((s) => s.session_id));
      setPersisted(list.filter((p) => !activeIds.has(p.session_id)));
    } catch (err) {
      console.error("Failed to list persisted sessions:", err);
    }
  }

  async function handleDelete(e: React.MouseEvent, sessionId: string) {
    e.stopPropagation();
    try {
      await invoke("delete_session", { sessionId });
      onDelete(sessionId);
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  }

  async function handleResume(e: React.MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (!apiKey.trim()) {
      alert("Please set an API key first");
      return;
    }
    try {
      const info = await invoke<SessionInfo>("resume_session", {
        apiKey: apiKey.trim(),
        model,
        budget,
        sessionId,
      });
      onResume(info);
    } catch (err) {
      console.error("Failed to resume session:", err);
    }
  }

  async function handleFork(e: React.MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (!apiKey.trim()) {
      alert("Please set an API key first");
      return;
    }
    try {
      const info = await invoke<SessionInfo>("fork_session", {
        apiKey: apiKey.trim(),
        model,
        budget,
        sourceSessionId: sessionId,
      });
      onFork(info);
    } catch (err) {
      console.error("Failed to fork session:", err);
    }
  }

  return (
    <div className="sidebar">
      <h2>Sessions</h2>

      {sessions.length === 0 && persisted.length === 0 && (
        <div style={{ fontSize: 13, color: "#5c6e8a" }}>No sessions yet</div>
      )}

      {sessions.map((s) => {
        const displayName = s.name || s.session_id.slice(0, 8) + "...";
        return (
          <button
            key={s.session_id}
            className={`session-item ${s.session_id === activeSessionId ? "active" : ""}`}
            onClick={() => onSelect(s.session_id)}
          >
            <div className="session-item-info">
              <div className="session-name" title={s.name || s.session_id}>
                {displayName}
              </div>
              <div className="session-meta">
                <span className="session-model">{s.model}</span>
                <span className="session-timestamp">
                  {formatTimestamp(s.created_at)}
                </span>
              </div>
            </div>
            <div className="session-actions">
              <button
                className="session-action-btn"
                title="Fork session"
                onClick={(e) => handleFork(e, s.session_id)}
              >
                ⑂
              </button>
              <button
                className="session-delete"
                title="Delete session"
                onClick={(e) => handleDelete(e, s.session_id)}
              >
                ✕
              </button>
            </div>
          </button>
        );
      })}

      {persisted.length > 0 && (
        <>
          <button
            className="persisted-toggle"
            onClick={() => setShowPersisted(!showPersisted)}
          >
            {showPersisted ? "▾" : "▸"} Past Sessions ({persisted.length})
          </button>

          {showPersisted &&
            persisted.map((p) => {
              const displayName =
                p.name || p.session_id.slice(0, 8) + "...";
              return (
                <button
                  key={p.session_id}
                  className="session-item persisted"
                  onClick={(e) => handleResume(e, p.session_id)}
                >
                  <div className="session-item-info">
                    <div
                      className="session-name"
                      title={p.name || p.session_id}
                    >
                      {displayName}
                    </div>
                    <div className="session-meta">
                      <span className="session-timestamp">
                        {formatTimestamp(p.created_at)}
                      </span>
                    </div>
                  </div>
                  <div className="session-actions">
                    <button
                      className="session-action-btn"
                      title="Resume session"
                      onClick={(e) => handleResume(e, p.session_id)}
                    >
                      ↻
                    </button>
                    <button
                      className="session-action-btn"
                      title="Fork session"
                      onClick={(e) => handleFork(e, p.session_id)}
                    >
                      ⑂
                    </button>
                  </div>
                </button>
              );
            })}
        </>
      )}
    </div>
  );
}

export default SessionSidebar;
