import { useState, useRef, useEffect } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { StepInfo } from "./RecursionTree";
import type { Artifact } from "./ArtifactViewer";

interface ChatMessage {
  role: "user" | "assistant" | "error";
  content: string;
  timestamp: string;
}

interface ToolApproval {
  callId: string;
  toolName: string;
  arguments: Record<string, unknown>;
}

type StreamEvent =
  | { type: "TokenDelta"; data: string }
  | { type: "ToolCallStart"; data: string }
  | { type: "ToolCallEnd"; data: string }
  | {
    type: "ToolApprovalRequired";
    data: { call_id: string; tool_name: string; arguments: Record<string, unknown> };
  }
  | {
    type: "StepProgress";
    data: {
      step: number;
      state: string;
      model?: string;
      tokens_in?: number;
      tokens_out?: number;
      cost_usd?: number;
    };
  }
  | {
    type: "ArtifactCreated";
    data: {
      file_path: string;
      old_content: string | null;
      new_content: string;
    };
  }
  | { type: "AssistantMessage"; data: string }
  | { type: "Error"; data: string }
  | { type: "CostUpdate"; data: number };

interface Props {
  sessionId: string;
  onCostUpdate: (cost: number) => void;
  onStepProgress: (step: StepInfo) => void;
  onArtifact: (artifact: Artifact) => void;
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const codeString = String(children).replace(/\n$/, "");
          // Inline code (no language class, short content)
          if (!match) {
            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          }
          // Code block with syntax highlighting
          return (
            <SyntaxHighlighter
              style={oneDark}
              language={match[1]}
              PreTag="div"
              customStyle={{
                margin: "0.5em 0",
                borderRadius: "6px",
                fontSize: "0.85em",
              }}
            >
              {codeString}
            </SyntaxHighlighter>
          );
        },
        // Style tables
        table({ children }) {
          return <table className="md-table">{children}</table>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function ToolApprovalDialog({
  approval,
  sessionId,
  onResolved,
}: {
  approval: ToolApproval;
  sessionId: string;
  onResolved: () => void;
}) {
  const [responding, setResponding] = useState(false);

  async function handleApproval(approved: boolean, rememberSession: boolean) {
    setResponding(true);
    try {
      await invoke("approve_tool_call", {
        sessionId,
        callId: approval.callId,
        approved,
        rememberSession,
      });
    } catch (e) {
      console.error("Failed to send approval:", e);
    }
    onResolved();
  }

  const argsPreview = JSON.stringify(approval.arguments, null, 2);

  return (
    <div className="tool-approval">
      <div className="tool-approval-header">
        Tool requires approval: <strong>{approval.toolName}</strong>
      </div>
      <pre className="tool-approval-args">{argsPreview}</pre>
      <div className="tool-approval-actions">
        <button
          className="btn-approve"
          disabled={responding}
          onClick={() => handleApproval(true, false)}
        >
          Allow once
        </button>
        <button
          className="btn-approve-session"
          disabled={responding}
          onClick={() => handleApproval(true, true)}
        >
          Allow for session
        </button>
        <button
          className="btn-deny"
          disabled={responding}
          onClick={() => handleApproval(false, false)}
        >
          Deny
        </button>
      </div>
    </div>
  );
}

function ChatPanel({ sessionId, onCostUpdate, onStepProgress, onArtifact }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [pendingApproval, setPendingApproval] = useState<ToolApproval | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, pendingApproval]);

  // Reset messages when session changes
  useEffect(() => {
    setMessages([]);
    setStreamingContent("");
    setPendingApproval(null);
  }, [sessionId]);

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;

    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);
    setStreamingContent("");

    const onEvent = new Channel<StreamEvent>();
    onEvent.onmessage = (event: StreamEvent) => {
      if (event.type === "TokenDelta") {
        setStreamingContent((prev) => prev + event.data);
      } else if (event.type === "ToolCallStart") {
        setStreamingContent((prev) => prev + `\n\`[calling ${event.data}...]\``);
      } else if (event.type === "ToolCallEnd") {
        setStreamingContent((prev) => prev + ` \`[${event.data} done]\`\n`);
      } else if (event.type === "ToolApprovalRequired") {
        setPendingApproval({
          callId: event.data.call_id,
          toolName: event.data.tool_name,
          arguments: event.data.arguments,
        });
      } else if (event.type === "AssistantMessage") {
        setStreamingContent("");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: event.data,
            timestamp: new Date().toISOString(),
          },
        ]);
      } else if (event.type === "Error") {
        setStreamingContent("");
        setMessages((prev) => [
          ...prev,
          {
            role: "error",
            content: event.data,
            timestamp: new Date().toISOString(),
          },
        ]);
      } else if (event.type === "StepProgress") {
        onStepProgress({
          step: event.data.step,
          state: event.data.state,
          model: event.data.model,
          tokensIn: event.data.tokens_in,
          tokensOut: event.data.tokens_out,
          costUsd: event.data.cost_usd,
          timestamp: Date.now(),
        });
      } else if (event.type === "ArtifactCreated") {
        onArtifact({
          filePath: event.data.file_path,
          oldContent: event.data.old_content,
          newContent: event.data.new_content,
          timestamp: Date.now(),
        });
      } else if (event.type === "CostUpdate") {
        onCostUpdate(event.data);
      }
    };

    try {
      await invoke("send_message", {
        sessionId,
        content: text,
        onEvent,
      });
    } catch (e) {
      setStreamingContent("");
      setMessages((prev) => [
        ...prev,
        {
          role: "error",
          content: String(e),
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
      setPendingApproval(null);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content">
              {msg.role === "assistant" ? (
                <MarkdownContent content={msg.content} />
              ) : (
                <div>{msg.content}</div>
              )}
            </div>
            <div className="timestamp">
              {new Date(msg.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        {streamingContent && (
          <div className="message assistant streaming">
            <div className="message-content">
              <MarkdownContent content={streamingContent} />
            </div>
          </div>
        )}
        {pendingApproval && (
          <ToolApprovalDialog
            approval={pendingApproval}
            sessionId={sessionId}
            onResolved={() => setPendingApproval(null)}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={sending}
        />
        <button onClick={handleSend} disabled={sending || !input.trim()}>
          {sending ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;
