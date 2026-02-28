import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

interface PromptTemplate {
  name: string;
  content: string;
}

interface Props {
  sessionId: string;
}

function SystemPromptEditor({ sessionId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    invoke<string>("get_system_prompt", { sessionId }).then(setPrompt);
    invoke<PromptTemplate[]>("list_prompt_templates").then(setTemplates);
  }, [sessionId]);

  async function handleSave() {
    setSaving(true);
    try {
      await invoke("update_system_prompt", { sessionId, prompt });
      setDirty(false);
    } catch (e) {
      console.error("Failed to update system prompt:", e);
    } finally {
      setSaving(false);
    }
  }

  function handleTemplateChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const template = templates.find((t) => t.name === e.target.value);
    if (template) {
      setPrompt(template.content);
      setDirty(true);
    }
  }

  const firstLine = prompt.split("\n")[0] || "System prompt";

  return (
    <div className="system-prompt-editor">
      <button
        className="system-prompt-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="collapse-icon">{expanded ? "\u25be" : "\u25b8"}</span>
        <span className="system-prompt-preview">{firstLine}</span>
      </button>
      {expanded && (
        <div className="system-prompt-body">
          <div className="system-prompt-controls">
            <select onChange={handleTemplateChange} defaultValue="">
              <option value="" disabled>
                Load template...
              </option>
              {templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
            <button
              onClick={handleSave}
              disabled={saving || !dirty}
              className="btn-save-prompt"
            >
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
          <textarea
            className="system-prompt-textarea"
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              setDirty(true);
            }}
            rows={6}
          />
        </div>
      )}
    </div>
  );
}

export default SystemPromptEditor;
