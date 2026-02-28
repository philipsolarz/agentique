import { invoke } from "@tauri-apps/api/core";

interface SessionInfo {
  session_id: string;
  model: string;
  budget: number;
  name: string;
  created_at: string;
}

interface Props {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
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

function SessionSidebar({ sessions, activeSessionId, onSelect, onDelete }: Props) {
  async function handleDelete(e: React.MouseEvent, sessionId: string) {
    e.stopPropagation();
    try {
      await invoke("delete_session", { sessionId });
      onDelete(sessionId);
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  }

  return (
    <div className="sidebar">
      <h2>Sessions</h2>
      {sessions.length === 0 && (
        <div style={{ fontSize: 13, color: "#5c6e8a" }}>
          No sessions yet
        </div>
      )}
      {sessions.map((s) => {
        const displayName =
          s.name || s.session_id.slice(0, 8) + "...";
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
            <button
              className="session-delete"
              title="Delete session"
              onClick={(e) => handleDelete(e, s.session_id)}
            >
              ✕
            </button>
          </button>
        );
      })}
    </div>
  );
}

export default SessionSidebar;
