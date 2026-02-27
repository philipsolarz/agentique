use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::types::{CompletionRequest, CompletionResponse, ProviderCapabilities, StreamChunk};

#[async_trait]
pub trait CompletionProvider: Send + Sync {
    /// Non-streaming completion.
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;

    /// Streaming completion returning a stream of chunks.
    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError>;

    /// Provider capabilities.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Provider name for logging.
    fn name(&self) -> &str;
}
