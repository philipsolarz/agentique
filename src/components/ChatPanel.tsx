import { useState, useRef, useEffect } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";

interface ChatMessage {
  role: "user" | "assistant" | "error";
  content: string;
  timestamp: string;
}

type StreamEvent =
  | { type: "TokenDelta"; data: string }
  | { type: "ToolCallStart"; data: string }
  | { type: "ToolCallEnd"; data: string }
  | { type: "AssistantMessage"; data: string }
  | { type: "Error"; data: string }
  | { type: "CostUpdate"; data: number };

interface Props {
  sessionId: string;
  onCostUpdate: (cost: number) => void;
}

function ChatPanel({ sessionId, onCostUpdate }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Reset messages when session changes
  useEffect(() => {
    setMessages([]);
    setStreamingContent("");
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
        setStreamingContent((prev) => prev + `\n[calling ${event.data}...]`);
      } else if (event.type === "ToolCallEnd") {
        setStreamingContent((prev) => prev + ` [${event.data} done]\n`);
      } else if (event.type === "AssistantMessage") {
        // Final message replaces streaming content
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
            <div>{msg.content}</div>
            <div className="timestamp">
              {new Date(msg.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        {streamingContent && (
          <div className="message assistant streaming">
            <div>{streamingContent}</div>
          </div>
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
