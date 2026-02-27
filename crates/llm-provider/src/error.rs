use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("API request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API returned error status {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Missing API key")]
    MissingApiKey,

    #[error("Rate limited, retry after {retry_after_secs:?}s")]
    RateLimited { retry_after_secs: Option<u64> },

    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error("{0}")]
    Other(String),
}
