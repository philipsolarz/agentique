use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    LlmCall,
    ToolExecution,
    UserInput,
    AgentResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTrace {
    pub id: Uuid,
    pub step_type: StepType,
    pub status: StepStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub input_summary: String,
    pub output_summary: Option<String>,
    pub error: Option<String>,
    pub token_usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl StepTrace {
    pub fn new(step_type: StepType, input_summary: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            step_type,
            status: StepStatus::Pending,
            started_at: Utc::now(),
            completed_at: None,
            input_summary: input_summary.into(),
            output_summary: None,
            error: None,
            token_usage: None,
        }
    }

    pub fn start(&mut self) {
        self.status = StepStatus::Running;
    }

    pub fn complete(&mut self, output_summary: impl Into<String>) {
        self.status = StepStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.output_summary = Some(output_summary.into());
    }

    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = StepStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error.into());
    }
}
