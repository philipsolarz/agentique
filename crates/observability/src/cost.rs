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

    /// Create a tracker with a ceiling in microdollars.
    pub fn with_microdollar_ceiling(microdollars: u64) -> Self {
        Self::new(microdollars)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_tracker_records_costs() {
        let tracker = BudgetTracker::new(1_000_000); // $1.00
        assert_eq!(tracker.spent_microdollars(), 0);
        assert_eq!(tracker.remaining_microdollars(), 1_000_000);

        let result = tracker.record(500_000);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 500_000);
        assert_eq!(tracker.spent_microdollars(), 500_000);
        assert_eq!(tracker.remaining_microdollars(), 500_000);
    }

    #[test]
    fn budget_tracker_rejects_on_exceeded() {
        let tracker = BudgetTracker::new(100);
        tracker.record(80).unwrap();
        let result = tracker.record(50); // total=130 > 100
        assert!(result.is_err());
    }

    #[test]
    fn budget_tracker_exact_ceiling_ok() {
        let tracker = BudgetTracker::new(100);
        let result = tracker.record(100); // exactly at ceiling
        assert!(result.is_ok());
    }

    #[test]
    fn budget_tracker_with_dollar_ceiling() {
        let tracker = BudgetTracker::with_dollar_ceiling(2.50);
        assert_eq!(tracker.remaining_microdollars(), 2_500_000);
        tracker.record(1_000_000).unwrap();
        assert!((tracker.spent_dollars() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_tracker_multiple_records() {
        let tracker = BudgetTracker::new(1000);
        for _ in 0..10 {
            tracker.record(100).unwrap();
        }
        assert_eq!(tracker.spent_microdollars(), 1000);
        assert!(tracker.record(1).is_err());
    }
}
