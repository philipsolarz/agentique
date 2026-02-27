interface SessionInfo {
  session_id: string;
  model: string;
  budget: number;
}

interface Props {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
}

function SessionSidebar({ sessions, activeSessionId, onSelect }: Props) {
  return (
    <div className="sidebar">
      <h2>Sessions</h2>
      {sessions.length === 0 && (
        <div style={{ fontSize: 13, color: "#5c6e8a" }}>
          No sessions yet
        </div>
      )}
      {sessions.map((s) => (
        <button
          key={s.session_id}
          className={`session-item ${s.session_id === activeSessionId ? "active" : ""}`}
          onClick={() => onSelect(s.session_id)}
        >
          <div>{s.session_id.slice(0, 8)}...</div>
          <div className="session-model">{s.model}</div>
        </button>
      ))}
    </div>
  );
}

export default SessionSidebar;
