pub mod executor;
pub mod recursion;
pub mod repl;
pub mod session;

pub use executor::{RippleExecutor, ToolDispatch};
pub use recursion::{
    BudgetPool, ExecutionWave, LoopDetector, SubAgentConfig, SubAgentResult, SubRlmCall,
    schedule_sub_agents,
};
pub use repl::{ReplSession, SearchMatch, Variable, VariableMetadata, SYMBOLIC_THRESHOLD_CHARS};
pub use session::{RippleConfig, RippleResult};
