use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
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
    base_url: Option<String>,
    custom_capabilities: Option<ProviderCapabilities>,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: "gpt-4o".to_string(),
            base_url: None,
            custom_capabilities: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set a custom base URL for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, etc.).
    /// The URL should be the full chat completions endpoint, e.g. `http://localhost:11434/v1/chat/completions`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Override provider capabilities (context window, pricing) for custom endpoints.
    pub fn with_capabilities(mut self, caps: ProviderCapabilities) -> Self {
        self.custom_capabilities = Some(caps);
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
    #[allow(dead_code)]
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

        let url = self.base_url.as_deref().unwrap_or(OPENAI_API_URL);
        let response = self
            .client
            .post(url)
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

        let url = self.base_url.as_deref().unwrap_or(OPENAI_API_URL);
        let response = self
            .client
            .post(url)
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

        let sse_stream = crate::sse::parse_sse_events::<StreamingResponse>(byte_stream)
            .filter_map(|result| async move {
                match result {
                    Err(e) => Some(Err(e)),
                    Ok(sr) => parse_streaming_chunk(sr).map(Ok),
                }
            });

        Ok(sse_stream.boxed())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        if let Some(caps) = &self.custom_capabilities {
            return caps.clone();
        }
        ProviderCapabilities {
            supports_streaming: true,
            supports_tool_calling: true,
            max_context_tokens: 128_000,
            cost_per_input_token: 2.50 / 1_000_000.0,
            cost_per_output_token: 10.0 / 1_000_000.0,
        }
    }

    fn name(&self) -> &str {
        if self.base_url.is_some() {
            "openai-compatible"
        } else {
            "openai"
        }
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
