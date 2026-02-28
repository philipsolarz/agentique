import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

interface AppSettings {
  api_key: string;
  model: string;
  budget: number | null;
  base_url: string | null;
}

interface Props {
  onSettingsLoaded: (settings: AppSettings) => void;
  apiKey: string;
  model: string;
  budget: string;
  baseUrl: string;
  onApiKeyChange: (v: string) => void;
  onModelChange: (v: string) => void;
  onBudgetChange: (v: string) => void;
  onBaseUrlChange: (v: string) => void;
}

function Settings({
  onSettingsLoaded,
  apiKey,
  model,
  budget,
  baseUrl,
  onApiKeyChange,
  onModelChange,
  onBudgetChange,
  onBaseUrlChange,
}: Props) {
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!loaded) {
      loadSettings();
    }
  }, [loaded]);

  async function loadSettings() {
    try {
      const settings = await invoke<AppSettings>("load_settings");
      onSettingsLoaded(settings);
      setLoaded(true);
    } catch (err) {
      console.error("Failed to load settings:", err);
      setLoaded(true);
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      await invoke("save_settings", {
        settings: {
          api_key: apiKey,
          model,
          budget: parseFloat(budget) || null,
          base_url: baseUrl.trim() || null,
        },
      });
    } catch (err) {
      console.error("Failed to save settings:", err);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="settings-panel">
      <h3>Settings</h3>
      <label>
        API Key
        <input
          type="password"
          value={apiKey}
          onChange={(e) => onApiKeyChange(e.target.value)}
          placeholder="sk-..."
        />
      </label>
      <label>
        Model
        <input
          type="text"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        />
      </label>
      <label>
        Budget (USD)
        <input
          type="text"
          value={budget}
          onChange={(e) => onBudgetChange(e.target.value)}
        />
      </label>
      <label>
        Base URL (optional, for Ollama/vLLM/LM Studio)
        <input
          type="text"
          value={baseUrl}
          onChange={(e) => onBaseUrlChange(e.target.value)}
          placeholder="http://localhost:11434/v1/chat/completions"
        />
      </label>
      <button onClick={handleSave} disabled={saving}>
        {saving ? "Saving..." : "Save Settings"}
      </button>
    </div>
  );
}

export default Settings;
