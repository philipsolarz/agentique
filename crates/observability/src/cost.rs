use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("budget exceeded: spent {spent_microdollars} microdollars, ceiling is {ceiling_microdollars}")]
    BudgetExceeded {
        spent_microdollars: u64,
        ceiling_microdollars: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cost_microdollars: u64,
    pub timestamp: DateTime<Utc>,
}

/// Lock-free budget tracker using atomic operations.
/// All costs are tracked in microdollars (1 USD = 1_000_000 microdollars).
#[derive(Debug)]
pub struct BudgetTracker {
    spent_microdollars: AtomicU64,
    ceiling_microdollars: u64,
}

impl BudgetTracker {
    /// Create a new tracker with a ceiling in microdollars.
    pub fn new(ceiling_microdollars: u64) -> Self {
        Self {
            spent_microdollars: AtomicU64::new(0),
            ceiling_microdollars,
        }
    }

    /// Create a tracker with a ceiling in dollars.
    pub fn with_dollar_ceiling(dollars: f64) -> Self {
        Self::new((dollars * 1_000_000.0) as u64)
    }

    /// Record a cost. Returns error if the budget ceiling would be exceeded.
    pub fn record(&self, cost_microdollars: u64) -> Result<u64, BudgetError> {
        let new_total = self
            .spent_microdollars
            .fetch_add(cost_microdollars, Ordering::Relaxed)
            + cost_microdollars;

        if new_total > self.ceiling_microdollars {
            Err(BudgetError::BudgetExceeded {
                spent_microdollars: new_total,
                ceiling_microdollars: self.ceiling_microdollars,
            })
        } else {
            Ok(new_total)
        }
    }

    pub fn spent_microdollars(&self) -> u64 {
        self.spent_microdollars.load(Ordering::Relaxed)
    }

    pub fn spent_dollars(&self) -> f64 {
        self.spent_microdollars() as f64 / 1_000_000.0
    }

    pub fn remaining_microdollars(&self) -> u64 {
        self.ceiling_microdollars
            .saturating_sub(self.spent_microdollars())
    }
}
