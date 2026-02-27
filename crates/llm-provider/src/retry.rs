use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::{info, warn};

use crate::error::ProviderError;
use crate::traits::CompletionProvider;
use crate::types::{CompletionRequest, CompletionResponse, ProviderCapabilities, StreamChunk};

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial backoff duration.
    pub initial_backoff: Duration,
    /// Maximum backoff duration.
    pub max_backoff: Duration,
    /// Jitter factor (0.0 to 1.0). Applied as random ±jitter% of the computed delay.
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            jitter_factor: 0.25,
        }
    }
}

/// A wrapper around a CompletionProvider that adds retry logic.
pub struct RetryProvider {
    inner: Box<dyn CompletionProvider>,
    config: RetryConfig,
}

impl RetryProvider {
    pub fn new(inner: Box<dyn CompletionProvider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    pub fn with_defaults(inner: Box<dyn CompletionProvider>) -> Self {
        Self::new(inner, RetryConfig::default())
    }

    fn is_retryable(error: &ProviderError) -> bool {
        match error {
            ProviderError::RateLimited { .. } => true,
            ProviderError::ApiError { status, .. } => {
                matches!(status, 500 | 502 | 503 | 529)
            }
            ProviderError::RequestFailed(_) => true,
            _ => false,
        }
    }

    fn compute_delay(&self, attempt: u32) -> Duration {
        let base = self.config.initial_backoff.as_millis() as f64 * 2.0_f64.powi(attempt as i32);
        let capped = base.min(self.config.max_backoff.as_millis() as f64);

        // Simple deterministic jitter: alternate between adding/subtracting
        let jitter_amount = capped * self.config.jitter_factor;
        let jittered = if attempt % 2 == 0 {
            capped + jitter_amount * 0.5
        } else {
            capped - jitter_amount * 0.5
        };

        Duration::from_millis(jittered.max(100.0) as u64)
    }
}

#[async_trait]
impl CompletionProvider for RetryProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.compute_delay(attempt - 1);
                info!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "Retrying after error"
                );
                tokio::time::sleep(delay).await;
            }

            match self.inner.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if Self::is_retryable(&e) && attempt < self.config.max_retries {
                        warn!(
                            attempt,
                            max_retries = self.config.max_retries,
                            error = %e,
                            "Retryable error, will retry"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ProviderError::Other("Max retries exhausted".to_string())))
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        // For streaming, we only retry the initial connection, not mid-stream errors
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.compute_delay(attempt - 1);
                info!(attempt, delay_ms = delay.as_millis() as u64, "Retrying stream");
                tokio::time::sleep(delay).await;
            }

            match self.inner.complete_streaming(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if Self::is_retryable(&e) && attempt < self.config.max_retries {
                        warn!(attempt, error = %e, "Retryable stream error, will retry");
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ProviderError::Other("Max retries exhausted".to_string())))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}
