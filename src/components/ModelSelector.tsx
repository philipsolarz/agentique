import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

const KNOWN_MODELS = [
  "claude-sonnet-4-20250514",
  "claude-opus-4-20250514",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
];

interface Props {
  sessionId: string;
  currentModel: string;
  disabled?: boolean;
  onModelSwitched: (model: string) => void;
}

function ModelSelector({ sessionId, currentModel, disabled, onModelSwitched }: Props) {
  const [customModel, setCustomModel] = useState("");
  const [showCustom, setShowCustom] = useState(false);
  const [switching, setSwitching] = useState(false);

  async function handleSwitch(model: string) {
    if (model === currentModel || switching) return;
    setSwitching(true);
    try {
      await invoke("switch_model", { sessionId, model });
      onModelSwitched(model);
    } catch (e) {
      console.error("Failed to switch model:", e);
    } finally {
      setSwitching(false);
    }
  }

  function handleSelectChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value;
    if (value === "__custom__") {
      setShowCustom(true);
    } else {
      setShowCustom(false);
      handleSwitch(value);
    }
  }

  function handleCustomSubmit() {
    if (customModel.trim()) {
      handleSwitch(customModel.trim());
      setShowCustom(false);
      setCustomModel("");
    }
  }

  return (
    <div className="model-selector">
      <select
        value={showCustom ? "__custom__" : currentModel}
        onChange={handleSelectChange}
        disabled={disabled || switching}
        className="model-select"
      >
        {!KNOWN_MODELS.includes(currentModel) && !showCustom && (
          <option value={currentModel}>{currentModel}</option>
        )}
        {KNOWN_MODELS.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
        <option value="__custom__">Custom...</option>
      </select>
      {showCustom && (
        <div className="model-custom-input">
          <input
            type="text"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            placeholder="model-name"
            onKeyDown={(e) => e.key === "Enter" && handleCustomSubmit()}
          />
          <button onClick={handleCustomSubmit} disabled={!customModel.trim()}>
            Go
          </button>
        </div>
      )}
    </div>
  );
}

export default ModelSelector;
