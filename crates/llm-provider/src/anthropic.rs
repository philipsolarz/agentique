use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::ProviderError;
use crate::traits::CompletionProvider;
use crate::types::*;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    default_model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: "claude-sonnet-4-20250514".to_string(),
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
        let mut system_text = None;
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    // Anthropic: system is a top-level field, not a message
                    system_text = msg.content.clone();
                }
                Role::User => {
                    messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Text(msg.content.clone().unwrap_or_default()),
                    });
                }
                Role::Assistant => {
                    if let Some(tool_calls) = &msg.tool_calls {
                        // Assistant message with tool use
                        let mut blocks = Vec::new();
                        if let Some(text) = &msg.content {
                            if !text.is_empty() {
                                blocks.push(ContentBlock::Text { text: text.clone() });
                            }
                        }
                        for tc in tool_calls {
                            let input: serde_json::Value =
                                serde_json::from_str(&tc.function.arguments)
                                    .unwrap_or(serde_json::Value::Object(Default::default()));
                            blocks.push(ContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                input,
                            });
                        }
                        messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Blocks(blocks),
                        });
                    } else {
                        messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Text(msg.content.clone().unwrap_or_default()),
                        });
                    }
                }
                Role::Tool => {
                    // Anthropic: tool results are user messages with tool_result content blocks
                    let block = ContentBlock::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: msg.content.clone().unwrap_or_default(),
                    };
                    // If the last message is already a user message with blocks, append to it.
                    // Otherwise create a new user message. Anthropic requires alternating roles.
                    if let Some(last) = messages.last_mut() {
                        if last.role == "user" {
                            match &mut last.content {
                                ApiContent::Blocks(blocks) => {
                                    blocks.push(block);
                                    continue;
                                }
                                ApiContent::Text(text) => {
                                    let mut blocks = vec![ContentBlock::Text {
                                        text: text.clone(),
                                    }];
                                    blocks.push(block);
                                    last.content = ApiContent::Blocks(blocks);
                                    continue;
                                }
                            }
                        }
                    }
                    messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Blocks(vec![block]),
                    });
                }
            }
        }

        // Convert tool definitions from OpenAI format to Anthropic format
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|td| AnthropicTool {
                    name: td.function.name.clone(),
                    description: td.function.description.clone(),
                    input_schema: td.function.parameters.clone(),
                })
                .collect()
        });

        ApiRequest {
            model: self.resolve_model(&request.model),
            system: system_text,
            messages,
            tools,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            stream: if stream { Some(true) } else { None },
        }
    }
}

// --- Anthropic API types ---

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

// --- Non-streaming response ---

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ResponseContentBlock>,
    stop_reason: Option<String>,
    usage: ApiUsage,
    model: String,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type")]
enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// --- Streaming event types ---

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStartData,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<DeltaUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: StreamErrorData },
}

#[derive(Deserialize, Debug)]
struct MessageStartData {
    usage: Option<ApiUsage>,
    #[allow(dead_code)]
    model: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentBlockStartData {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize, Debug)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct DeltaUsage {
    output_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct StreamErrorData {
    message: String,
}

fn parse_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some(other) => FinishReason::Unknown(other.to_string()),
        None => FinishReason::Unknown("none".to_string()),
    }
}

#[async_trait]
impl CompletionProvider for AnthropicProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let model = self.resolve_model(&request.model);
        let api_request = self.build_api_request(&request, false);

        debug!(model = %model, messages = request.messages.len(), "Sending Anthropic completion request");

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                warn!("Rate limited by Anthropic");
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

        // Convert response content blocks to our Message type
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in &api_response.content {
            match block {
                ResponseContentBlock::Text { text } => {
                    text_parts.push(text.clone());
                }
                ResponseContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
            }
        }

        let content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        let message = Message {
            role: Role::Assistant,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        };

        let usage = Usage {
            prompt_tokens: api_response.usage.input_tokens,
            completion_tokens: api_response.usage.output_tokens,
            total_tokens: api_response.usage.input_tokens + api_response.usage.output_tokens,
        };

        let finish_reason = parse_stop_reason(api_response.stop_reason.as_deref());

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

        debug!(model = %model, messages = request.messages.len(), "Sending Anthropic streaming request");

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
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

        // Track active tool call state across SSE events
        // Anthropic streams tool call arguments as input_json_delta chunks
        // that need to be assembled and emitted as ToolCall deltas
        let sse_stream = byte_stream
            .map(|result| match result {
                Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).to_string()),
                Err(e) => Err(ProviderError::RequestFailed(e)),
            })
            .scan(
                StreamState {
                    buffer: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                },
                |state, chunk_result| {
                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => return futures::future::ready(Some(vec![Err(e)])),
                    };
                    state.buffer.push_str(&chunk);

                    let mut events = Vec::new();
                    while let Some(pos) = state.buffer.find("\n\n") {
                        let event_text = state.buffer[..pos].to_string();
                        state.buffer = state.buffer[pos + 2..].to_string();

                        // Parse SSE: may have "event: ..." and "data: ..." lines
                        let mut data_line = None;
                        for line in event_text.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                data_line = Some(data.trim().to_string());
                            }
                        }

                        let Some(data) = data_line else {
                            continue;
                        };

                        match serde_json::from_str::<StreamEvent>(&data) {
                            Ok(event) => {
                                if let Some(chunk) = process_stream_event(event, state) {
                                    events.push(Ok(chunk));
                                }
                            }
                            Err(e) => {
                                // Some events we don't handle; only error on real parse failures
                                debug!(error = %e, data = %data, "Skipping unparseable SSE event");
                            }
                        }
                    }

                    futures::future::ready(Some(events))
                },
            )
            .flat_map(stream::iter);

        Ok(sse_stream.boxed())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Claude Sonnet 4 pricing: $3/M input, $15/M output, 200K context
        ProviderCapabilities {
            supports_streaming: true,
            supports_tool_calling: true,
            max_context_tokens: 200_000,
            cost_per_input_token: 3.0 / 1_000_000.0,
            cost_per_output_token: 15.0 / 1_000_000.0,
        }
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

struct StreamState {
    buffer: String,
    input_tokens: u32,
    output_tokens: u32,
}

fn process_stream_event(event: StreamEvent, state: &mut StreamState) -> Option<StreamChunk> {
    match event {
        StreamEvent::MessageStart { message } => {
            if let Some(usage) = message.usage {
                state.input_tokens = usage.input_tokens;
            }
            None
        }
        StreamEvent::ContentBlockStart {
            content_block: ContentBlockStartData::ToolUse { id, name },
            ..
        } => Some(StreamChunk {
            delta_content: None,
            delta_tool_calls: Some(vec![ToolCall {
                id,
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name,
                    arguments: String::new(),
                },
            }]),
            finish_reason: None,
            usage: None,
        }),
        StreamEvent::ContentBlockStart {
            content_block: ContentBlockStartData::Text { text },
            ..
        } => {
            if text.is_empty() {
                None
            } else {
                Some(StreamChunk {
                    delta_content: Some(text),
                    delta_tool_calls: None,
                    finish_reason: None,
                    usage: None,
                })
            }
        }
        StreamEvent::ContentBlockDelta { delta, .. } => match delta {
            ContentDelta::TextDelta { text } => Some(StreamChunk {
                delta_content: Some(text),
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
            }),
            ContentDelta::InputJsonDelta { partial_json } => Some(StreamChunk {
                delta_content: None,
                delta_tool_calls: Some(vec![ToolCall {
                    id: String::new(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: String::new(),
                        arguments: partial_json,
                    },
                }]),
                finish_reason: None,
                usage: None,
            }),
        },
        StreamEvent::MessageDelta { delta, usage } => {
            if let Some(u) = usage {
                state.output_tokens += u.output_tokens;
            }
            let finish_reason = delta
                .stop_reason
                .as_deref()
                .map(|r| parse_stop_reason(Some(r)));
            if finish_reason.is_some() {
                Some(StreamChunk {
                    delta_content: None,
                    delta_tool_calls: None,
                    finish_reason,
                    usage: Some(Usage {
                        prompt_tokens: state.input_tokens,
                        completion_tokens: state.output_tokens,
                        total_tokens: state.input_tokens + state.output_tokens,
                    }),
                })
            } else {
                None
            }
        }
        StreamEvent::Error { error } => {
            warn!(message = %error.message, "Anthropic stream error");
            None
        }
        _ => None,
    }
}
