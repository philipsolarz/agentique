use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::repl::Variable;

/// Configuration for sub-agent spawning and budget allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentConfig {
    /// Maximum recursion depth (0 = no sub-agents allowed).
    pub max_depth: u8,
    /// Fraction of parent's remaining budget allocated to each sub-agent.
    pub budget_fraction: f64,
    /// Minimum budget in microdollars for a sub-agent to be viable.
    pub min_viable_budget_microdollars: u64,
    /// Maximum budget per sub-agent in microdollars, regardless of fraction.
    pub max_sub_budget_microdollars: u64,
    /// Maximum number of concurrent sub-agents (0 = unlimited).
    pub max_concurrent: usize,
}

impl Default for SubAgentConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            budget_fraction: 0.15,
            min_viable_budget_microdollars: 10_000, // $0.01
            max_sub_budget_microdollars: 500_000,   // $0.50
            max_concurrent: 8,
        }
    }
}

impl SubAgentConfig {
    /// Calculate the budget allocation for a sub-agent.
    pub fn allocate_budget(&self, remaining_microdollars: u64) -> Option<u64> {
        let allocated =
            ((remaining_microdollars as f64) * self.budget_fraction) as u64;
        let capped = allocated.min(self.max_sub_budget_microdollars);
        if capped < self.min_viable_budget_microdollars {
            None
        } else {
            Some(capped)
        }
    }
}

/// Tracks sub-agent calls to detect loops (same instruction + data at same depth).
#[derive(Debug)]
pub struct LoopDetector {
    seen: HashSet<u64>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self {
            seen: HashSet::new(),
        }
    }

    /// Check if this (instruction, data_preview, depth) combination has been seen before.
    /// Returns true if it's a duplicate (loop detected).
    pub fn check_and_record(&mut self, instruction: &str, data_preview: &str, depth: u8) -> bool {
        let mut hasher = std::hash::DefaultHasher::new();
        instruction.hash(&mut hasher);
        data_preview.hash(&mut hasher);
        depth.hash(&mut hasher);
        let hash = hasher.finish();
        !self.seen.insert(hash)
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared budget pool for concurrent sub-agents.
/// Uses atomic operations for lock-free budget management.
#[derive(Debug)]
pub struct BudgetPool {
    remaining_microdollars: AtomicU64,
}

impl BudgetPool {
    pub fn new(total_microdollars: u64) -> Self {
        Self {
            remaining_microdollars: AtomicU64::new(total_microdollars),
        }
    }

    /// Try to reserve `amount` microdollars from the pool.
    /// Returns the actual amount reserved (may be less if pool is nearly empty).
    /// Returns None if the pool is exhausted below the minimum viable amount.
    pub fn try_reserve(&self, amount: u64, min_viable: u64) -> Option<u64> {
        loop {
            let current = self.remaining_microdollars.load(Ordering::Relaxed);
            if current < min_viable {
                return None;
            }
            let to_reserve = amount.min(current);
            if to_reserve < min_viable {
                return None;
            }
            match self.remaining_microdollars.compare_exchange_weak(
                current,
                current - to_reserve,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(to_reserve),
                Err(_) => continue, // CAS failed, retry
            }
        }
    }

    /// Return unused budget back to the pool.
    pub fn return_unused(&self, amount: u64) {
        self.remaining_microdollars
            .fetch_add(amount, Ordering::Release);
    }

    pub fn remaining(&self) -> u64 {
        self.remaining_microdollars.load(Ordering::Relaxed)
    }
}

/// Context passed to a sub-agent execution.
#[derive(Debug)]
pub struct SubAgentContext {
    pub depth: u8,
    pub instruction: String,
    pub data_variable_name: String,
    pub budget_microdollars: u64,
    pub config: SubAgentConfig,
}

/// Result from a sub-agent execution.
#[derive(Debug)]
pub struct SubAgentResult {
    pub output: Option<Variable>,
    pub cost_microdollars: u64,
    pub steps_taken: u32,
}

/// A parsed sub_rlm call extracted from tool calls, used for dependency analysis.
#[derive(Debug, Clone)]
pub struct SubRlmCall {
    /// Index in the original tool_calls array.
    pub index: usize,
    /// Tool call ID for correlating results.
    pub call_id: String,
    /// The REPL variable to read data from.
    pub variable_name: String,
    /// The instruction for the sub-agent.
    pub instruction: String,
    /// The REPL variable name to store the result.
    pub result_variable: String,
    /// The raw JSON arguments.
    pub args: serde_json::Value,
}

/// An execution wave: a group of sub-agent calls that can run concurrently.
/// Calls within a wave have no data dependencies on each other.
/// Waves must be executed sequentially (wave N's results may be needed by wave N+1).
#[derive(Debug)]
pub struct ExecutionWave {
    pub calls: Vec<SubRlmCall>,
}

/// Analyze a batch of sub_rlm calls and schedule them into execution waves.
///
/// Dependency rule: if call B's `variable_name` equals call A's `result_variable`,
/// then B depends on A and must run in a later wave.
///
/// Returns waves in execution order. Independent calls share a wave (concurrent dispatch).
pub fn schedule_sub_agents(calls: Vec<SubRlmCall>) -> Vec<ExecutionWave> {
    if calls.is_empty() {
        return vec![];
    }
    if calls.len() == 1 {
        return vec![ExecutionWave { calls }];
    }

    // Build a set of result variables produced by each call
    // and determine which calls depend on those results.
    let mut waves: Vec<ExecutionWave> = Vec::new();
    let mut scheduled: HashSet<usize> = HashSet::new();
    let mut available_results: HashSet<String> = HashSet::new();
    let total = calls.len();

    // Keep scheduling until all calls are placed in a wave
    while scheduled.len() < total {
        let mut current_wave = Vec::new();

        for call in &calls {
            if scheduled.contains(&call.index) {
                continue;
            }
            // Check if this call depends on a result_variable from an unscheduled call
            let depends_on_pending = calls.iter().any(|other| {
                other.index != call.index
                    && !scheduled.contains(&other.index)
                    && call.variable_name == other.result_variable
            });
            // Check if this call depends on a result from a call in the *current* wave
            let depends_on_current_wave = current_wave.iter().any(|w: &SubRlmCall| {
                call.variable_name == w.result_variable
            });
            // A call can be scheduled if it either:
            // - has no dependency on any pending result, OR
            // - depends on a result that's already available from a previous wave
            let input_available =
                available_results.contains(&call.variable_name) || !depends_on_pending;

            if input_available && !depends_on_current_wave {
                current_wave.push(call.clone());
            }
        }

        if current_wave.is_empty() {
            // Cycle detection / safety valve: force-schedule remaining calls sequentially
            for call in &calls {
                if !scheduled.contains(&call.index) {
                    let idx = call.index;
                    waves.push(ExecutionWave {
                        calls: vec![call.clone()],
                    });
                    scheduled.insert(idx);
                    available_results.insert(call.result_variable.clone());
                }
            }
            break;
        }

        for call in &current_wave {
            scheduled.insert(call.index);
            available_results.insert(call.result_variable.clone());
        }
        waves.push(ExecutionWave {
            calls: current_wave,
        });
    }

    waves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_allocation() {
        let config = SubAgentConfig::default();
        // 15% of $1.00 (1_000_000 microdollars) = $0.15 (150_000)
        assert_eq!(config.allocate_budget(1_000_000), Some(150_000));
        // Capped at max_sub_budget
        assert_eq!(config.allocate_budget(10_000_000), Some(500_000));
        // Below minimum viable
        assert_eq!(config.allocate_budget(50_000), None); // 15% of 50k = 7500 < 10k
    }

    #[test]
    fn loop_detection() {
        let mut detector = LoopDetector::new();
        assert!(!detector.check_and_record("find errors", "fn main()", 0));
        assert!(!detector.check_and_record("find errors", "fn main()", 1)); // different depth
        assert!(detector.check_and_record("find errors", "fn main()", 0)); // duplicate!
    }

    #[test]
    fn budget_pool_reservation() {
        let pool = BudgetPool::new(1_000_000);
        // Reserve 150k
        assert_eq!(pool.try_reserve(150_000, 10_000), Some(150_000));
        assert_eq!(pool.remaining(), 850_000);
        // Reserve more than remaining
        assert_eq!(pool.try_reserve(900_000, 10_000), Some(850_000));
        assert_eq!(pool.remaining(), 0);
        // Pool exhausted
        assert_eq!(pool.try_reserve(100, 10_000), None);
    }

    #[test]
    fn budget_pool_return() {
        let pool = BudgetPool::new(1_000_000);
        pool.try_reserve(500_000, 10_000);
        pool.return_unused(200_000);
        assert_eq!(pool.remaining(), 700_000);
    }

    // --- Async sub-agent scheduling tests ---

    fn make_call(index: usize, var_name: &str, result_var: &str) -> SubRlmCall {
        SubRlmCall {
            index,
            call_id: format!("call_{index}"),
            variable_name: var_name.to_string(),
            instruction: format!("process {var_name}"),
            result_variable: result_var.to_string(),
            args: serde_json::json!({}),
        }
    }

    #[test]
    fn schedule_empty() {
        let waves = schedule_sub_agents(vec![]);
        assert!(waves.is_empty());
    }

    #[test]
    fn schedule_single_call() {
        let calls = vec![make_call(0, "data", "result_0")];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].calls.len(), 1);
    }

    #[test]
    fn schedule_independent_calls_same_wave() {
        // Three calls all reading from different existing variables, no dependencies
        let calls = vec![
            make_call(0, "chunk_0", "result_0"),
            make_call(1, "chunk_1", "result_1"),
            make_call(2, "chunk_2", "result_2"),
        ];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 1, "All independent calls should be in one wave");
        assert_eq!(waves[0].calls.len(), 3);
    }

    #[test]
    fn schedule_chain_dependency() {
        // A produces result_a, B reads result_a → must be serialized
        let calls = vec![
            make_call(0, "data", "result_a"),
            make_call(1, "result_a", "result_b"),
        ];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 2, "Dependent calls need separate waves");
        assert_eq!(waves[0].calls.len(), 1);
        assert_eq!(waves[0].calls[0].index, 0);
        assert_eq!(waves[1].calls.len(), 1);
        assert_eq!(waves[1].calls[0].index, 1);
    }

    #[test]
    fn schedule_mixed_deps_and_independent() {
        // call 0: reads "data", writes "partial_0"
        // call 1: reads "data", writes "partial_1"   (independent of 0)
        // call 2: reads "partial_0", writes "merged"  (depends on 0)
        let calls = vec![
            make_call(0, "data", "partial_0"),
            make_call(1, "data", "partial_1"),
            make_call(2, "partial_0", "merged"),
        ];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 2);
        // Wave 0: calls 0 and 1 (both independent, read "data")
        assert_eq!(waves[0].calls.len(), 2);
        let wave0_indices: HashSet<usize> = waves[0].calls.iter().map(|c| c.index).collect();
        assert!(wave0_indices.contains(&0));
        assert!(wave0_indices.contains(&1));
        // Wave 1: call 2 (depends on call 0's result)
        assert_eq!(waves[1].calls.len(), 1);
        assert_eq!(waves[1].calls[0].index, 2);
    }

    #[test]
    fn schedule_three_level_chain() {
        // A → B → C
        let calls = vec![
            make_call(0, "data", "step_1"),
            make_call(1, "step_1", "step_2"),
            make_call(2, "step_2", "step_3"),
        ];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0].calls[0].index, 0);
        assert_eq!(waves[1].calls[0].index, 1);
        assert_eq!(waves[2].calls[0].index, 2);
    }

    #[test]
    fn schedule_fan_out_fan_in() {
        // Fan-out: 0,1,2 read "data", produce "r0","r1","r2"
        // Fan-in: 3 reads "r0" (depends on 0), but not r1 or r2
        let calls = vec![
            make_call(0, "data", "r0"),
            make_call(1, "data", "r1"),
            make_call(2, "data", "r2"),
            make_call(3, "r0", "merged"),
        ];
        let waves = schedule_sub_agents(calls);
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0].calls.len(), 3);
        assert_eq!(waves[1].calls.len(), 1);
        assert_eq!(waves[1].calls[0].index, 3);
    }

    #[test]
    fn budget_pool_concurrent_reservation() {
        use std::sync::Arc;

        let pool = Arc::new(BudgetPool::new(1_000_000));
        let mut handles = vec![];

        // Spawn 20 threads each trying to reserve 100k
        for _ in 0..20 {
            let pool = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                pool.try_reserve(100_000, 10_000)
            }));
        }

        let mut total_reserved = 0u64;
        let mut success_count = 0;
        for h in handles {
            if let Some(amount) = h.join().unwrap() {
                total_reserved += amount;
                success_count += 1;
            }
        }

        // 10 should succeed (10 * 100k = 1M), rest should fail
        assert_eq!(total_reserved, 1_000_000);
        assert_eq!(success_count, 10);
        assert_eq!(pool.remaining(), 0);
    }
}
