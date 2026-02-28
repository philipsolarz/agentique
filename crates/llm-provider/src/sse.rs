use futures::stream::{self, BoxStream, Stream, StreamExt};
use serde::de::DeserializeOwned;
use tracing::debug;

use crate::error::ProviderError;

/// Parse an SSE byte stream into typed events.
///
/// Handles:
/// - Buffering across chunk boundaries
/// - Splitting on `\n\n` event boundaries
/// - Extracting `data: ` prefixed lines
/// - Skipping `[DONE]` sentinel
/// - Deserializing each data line as JSON into type `T`
///
/// Events that fail to parse are logged and skipped.
pub fn parse_sse_events<T: DeserializeOwned + Send + 'static>(
    byte_stream: impl Stream<Item = Result<impl AsRef<[u8]>, reqwest::Error>> + Send + 'static,
) -> BoxStream<'static, Result<T, ProviderError>> {
    byte_stream
        .map(|result| match result {
            Ok(bytes) => Ok(String::from_utf8_lossy(bytes.as_ref()).to_string()),
            Err(e) => Err(ProviderError::RequestFailed(e)),
        })
        .scan(String::new(), |buffer: &mut String, chunk_result: Result<String, ProviderError>| {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => return futures::future::ready(Some(vec![Err(e)])),
            };
            buffer.push_str(&chunk);

            let mut events: Vec<Result<T, ProviderError>> = Vec::new();
            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                *buffer = buffer[pos + 2..].to_string();

                for line in event_text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        let data = data.trim();
                        if data == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<T>(data) {
                            Ok(event) => events.push(Ok(event)),
                            Err(e) => {
                                debug!(error = %e, data = %data, "Skipping unparseable SSE event");
                            }
                        }
                    }
                }
            }

            futures::future::ready(Some(events))
        })
        .flat_map(stream::iter)
        .boxed()
}
