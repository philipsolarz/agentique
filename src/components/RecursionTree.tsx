import { useState } from "react";

export interface StepInfo {
  step: number;
  state: string;
  model?: string;
  tokensIn?: number;
  tokensOut?: number;
  costUsd?: number;
  timestamp: number;
}

interface Props {
  steps: StepInfo[];
}

const STATE_COLORS: Record<string, string> = {
  llm_call: "#4a90d9",
  llm_done: "#2e7d32",
  tool_execution: "#f9a825",
  sub_agent: "#9c27b0",
  repl_eval: "#00897b",
  error: "#c62828",
};

function stateLabel(state: string): string {
  switch (state) {
    case "llm_call":
      return "LLM Call";
    case "llm_done":
      return "LLM Done";
    case "tool_execution":
      return "Tool Exec";
    case "sub_agent":
      return "Sub-Agent";
    case "repl_eval":
      return "REPL Eval";
    default:
      return state;
  }
}

function StepNode({ step }: { step: StepInfo }) {
  const color = STATE_COLORS[step.state] || "#7b8cad";
  const totalTokens =
    step.tokensIn != null && step.tokensOut != null
      ? step.tokensIn + step.tokensOut
      : null;

  return (
    <div className="step-node" style={{ borderLeftColor: color }}>
      <div className="step-header">
        <span className="step-badge" style={{ background: color }}>
          {stateLabel(step.state)}
        </span>
        <span className="step-number">#{step.step}</span>
        {step.model && <span className="step-model">{step.model}</span>}
      </div>
      {(totalTokens != null || step.costUsd != null) && (
        <div className="step-details">
          {totalTokens != null && (
            <span className="step-tokens">
              {step.tokensIn?.toLocaleString()}in / {step.tokensOut?.toLocaleString()}out
            </span>
          )}
          {step.costUsd != null && (
            <span className="step-cost">${step.costUsd.toFixed(4)}</span>
          )}
        </div>
      )}
    </div>
  );
}

function RecursionTree({ steps }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (steps.length === 0) return null;

  return (
    <div className="recursion-tree">
      <div
        className="recursion-tree-header"
        onClick={() => setCollapsed(!collapsed)}
      >
        <span className="collapse-icon">{collapsed ? "\u25B6" : "\u25BC"}</span>
        <span>Execution Steps ({steps.length})</span>
      </div>
      {!collapsed && (
        <div className="recursion-tree-body">
          {steps.map((s, i) => (
            <StepNode key={`${s.step}-${s.state}-${i}`} step={s} />
          ))}
        </div>
      )}
    </div>
  );
}

export default RecursionTree;
