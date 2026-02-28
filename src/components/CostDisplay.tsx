import { useState } from "react";
import type { StepInfo } from "./RecursionTree";

interface Props {
  cost: number;
  budget: number;
  steps: StepInfo[];
}

function CostDisplay({ cost, budget, steps }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = budget > 0 ? (cost / budget) * 100 : 0;
  const isWarning = pct >= 80 && pct < 100;
  const isExceeded = pct >= 100;

  const barColor = isExceeded ? "#c62828" : isWarning ? "#f9a825" : "#2e7d32";

  // Filter to completed steps that have cost info
  const completedSteps = steps.filter(
    (s) => s.state === "llm_done" && s.costUsd != null
  );

  return (
    <div className="cost-dashboard">
      <div
        className="cost-summary"
        onClick={() => setExpanded(!expanded)}
        title="Click to expand cost details"
      >
        <span className="cost-amount" style={{ color: isExceeded ? "#ff6b6b" : isWarning ? "#f9a825" : "#7b8cad" }}>
          ${cost.toFixed(4)}
        </span>
        <span className="cost-separator">/</span>
        <span className="cost-budget">${budget.toFixed(2)}</span>
        <div className="cost-bar">
          <div
            className="cost-bar-fill"
            style={{
              width: `${Math.min(pct, 100)}%`,
              background: barColor,
            }}
          />
        </div>
      </div>

      {expanded && completedSteps.length > 0 && (
        <div className="cost-breakdown">
          <table className="cost-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Model</th>
                <th>Tokens</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {completedSteps.map((s, i) => (
                <tr key={i}>
                  <td>#{s.step}</td>
                  <td>{s.model || "—"}</td>
                  <td>
                    {s.tokensIn != null
                      ? `${s.tokensIn.toLocaleString()} / ${s.tokensOut?.toLocaleString()}`
                      : "—"}
                  </td>
                  <td>${s.costUsd?.toFixed(4) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default CostDisplay;
