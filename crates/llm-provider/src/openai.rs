use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::ProviderError;
use crate::traits::CompletionProvider;
use crate::types::*;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    default_model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: "gpt-4o".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn resolve_model(&self, request_model: &str) -> String {
        if request_model.is_empty() {
            self.default_model.clone()
        } else {
            request_model.to_string()
        }
    }

    fn build_api_request(&self, request: &CompletionRequest, stream: bool) -> ApiRequest {
        ApiRequest {
            model: self.resolve_model(&request.model),
            messages: request.messages.iter().map(ApiMessage::from).collect(),
            tools: request.tools.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: if stream { Some(true) } else { None },
            stream_options: if stream {
                Some(StreamOptions { include_usage: true })
            } else {
                None
            },
        }
    }
}

// --- OpenAI API request/response types ---

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for ApiMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        ApiMessage {
            role: role.to_string(),
            content: msg.content.clone(),
            tool_calls: msg.tool_calls.clone(),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<ApiChoice>,
    usage: Option<ApiUsage>,
    model: String,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Deserialize, Clone)]
struct ApiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ApiToolCallFunction,
}

#[derive(Deserialize, Clone)]
struct ApiToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// --- SSE streaming types ---

#[derive(Deserialize)]
struct StreamingResponse {
    choices: Option<Vec<StreamingChoice>>,
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamingDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamingToolCall>>,
}

#[derive(Deserialize)]
struct StreamingToolCall {
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<StreamingToolCallFunction>,
}

#[derive(Deserialize)]
struct StreamingToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Unknown(other.to_string()),
        None => FinishReason::Unknown("none".to_string()),
    }
}

fn convert_tool_calls(tool_calls: Vec<ApiToolCall>) -> Vec<ToolCall> {
    tool_calls
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            call_type: tc.call_type,
            function: ToolCallFunction {
                name: tc.function.name,
                arguments: tc.function.arguments,
            },
        })
        .collect()
}

#[async_trait]
impl CompletionProvider for OpenAiProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let model = self.resolve_model(&request.model);
        let api_request = self.build_api_request(&request, false);

        debug!(model = %model, messages = request.messages.len(), "Sending completion request");

        let response = self
            .client
            .post(OPENAI_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&api_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                warn!("Rate limited by OpenAI");
                return Err(ProviderError::RateLimited {
                    retry_after_secs: None,
                });
            }
            return Err(ProviderError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let api_response: ApiResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::ParseError(e.to_string()))?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::ParseError("No choices in response".to_string()))?;

        let tool_calls = choice.message.tool_calls.map(convert_tool_calls);

        let message = Message {
            role: Role::Assistant,
            content: choice.message.content,
            tool_calls,
            tool_call_id: None,
        };

        let usage = api_response
            .usage
            .map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            })
            .unwrap_or_default();

        let finish_reason = parse_finish_reason(choice.finish_reason.as_deref());

        Ok(CompletionResponse {
            message,
            finish_reason,
            usage,
            model: api_response.model,
        })
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        let model = self.resolve_model(&request.model);
        let api_request = self.build_api_request(&request, true);

        debug!(model = %model, messages = request.messages.len(), "Sending streaming request");

        let response = self
            .client
            .post(OPENAI_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&api_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(ProviderError::RateLimited {
                    retry_after_secs: None,
                });
            }
            return Err(ProviderError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let byte_stream = response.bytes_stream();

        // Parse SSE events from the byte stream
        let sse_stream = byte_stream
            .map(|result| match result {
                Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).to_string()),
                Err(e) => Err(ProviderError::RequestFailed(e)),
            })
            // SSE data can arrive in chunks that span multiple events or partial events.
            // We accumulate a buffer and split on double-newline boundaries.
            .scan(String::new(), |buffer, chunk_result| {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => return futures::future::ready(Some(vec![Err(e)])),
                };
                buffer.push_str(&chunk);

                let mut events = Vec::new();
                while let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    *buffer = buffer[pos + 2..].to_string();

                    for line in event_text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            let data = data.trim();
                            if data == "[DONE]" {
                                continue;
                            }
                            match serde_json::from_str::<StreamingResponse>(data) {
                                Ok(sr) => {
                                    if let Some(chunk) = parse_streaming_chunk(sr) {
                                        events.push(Ok(chunk));
                                    }
                                }
                                Err(e) => {
                                    events.push(Err(ProviderError::ParseError(format!(
                                        "SSE parse error: {e}"
                                    ))));
                                }
                            }
                        }
                    }
                }

                futures::future::ready(Some(events))
            })
            .flat_map(stream::iter);

        Ok(sse_stream.boxed())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: true,
            supports_tool_calling: true,
            max_context_tokens: 128_000,
            cost_per_input_token: 2.50 / 1_000_000.0,
            cost_per_output_token: 10.0 / 1_000_000.0,
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

fn parse_streaming_chunk(sr: StreamingResponse) -> Option<StreamChunk> {
    // Handle usage-only chunks (sent at the end when stream_options.include_usage is set)
    if let Some(usage) = sr.usage {
        return Some(StreamChunk {
            delta_content: None,
            delta_tool_calls: None,
            finish_reason: None,
            usage: Some(Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            }),
        });
    }

    let choices = sr.choices?;
    let choice = choices.into_iter().next()?;

    let delta_content = choice.delta.content;

    let delta_tool_calls = choice.delta.tool_calls.map(|tcs| {
        tcs.into_iter()
            .map(|tc| ToolCall {
                id: tc.id.unwrap_or_default(),
                call_type: tc.call_type.unwrap_or_else(|| "function".to_string()),
                function: ToolCallFunction {
                    name: tc
                        .function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default(),
                    arguments: tc
                        .function
                        .as_ref()
                        .and_then(|f| f.arguments.clone())
                        .unwrap_or_default(),
                },
            })
            .collect()
    });

    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(|r| parse_finish_reason(Some(r)));

    Some(StreamChunk {
        delta_content,
        delta_tool_calls,
        finish_reason,
        usage: None,
    })
}
