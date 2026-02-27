use serde::{Deserialize, Serialize};

/// Configuration for a Ripple execution session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RippleConfig {
    /// Maximum agent loop steps before forced termination.
    pub max_steps: u32,
    /// Maximum budget in microdollars (1 USD = 1_000_000).
    pub max_budget_microdollars: u64,
    /// Default model to use.
    pub default_model: String,
}

impl Default for RippleConfig {
    fn default() -> Self {
        Self {
            max_steps: 50,
            max_budget_microdollars: 5_000_000, // $5
            default_model: "gpt-4o".to_string(),
        }
    }
}

/// The result of a completed Ripple execution.
#[derive(Debug)]
pub struct RippleResult {
    pub steps_taken: u32,
    pub total_cost_microdollars: u64,
    pub final_output: Option<String>,
}
