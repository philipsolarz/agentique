use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::{info, warn};

use crate::error::ProviderError;
use crate::traits::CompletionProvider;
use crate::types::{CompletionRequest, CompletionResponse, ProviderCapabilities, StreamChunk};

/// Model tier for depth-based routing.
/// Tier 1 = most capable (root agent), Tier 3 = cheapest (deep sub-agents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelTier {
    Tier1,
    Tier2,
    Tier3,
}

/// A registered provider with its tier assignment.
struct RegisteredProvider {
    provider: Box<dyn CompletionProvider>,
    tier: ModelTier,
}

/// Routes LLM requests to the appropriate provider based on configurable policies.
///
/// Supports:
/// - **Depth-based routing**: Maps recursion depth to model tier
/// - **Cost-based routing**: Selects cheapest provider meeting capability requirements
/// - **Fallback chain**: On retryable errors, tries the next provider in the chain
pub struct ModelRouter {
    providers: Vec<RegisteredProvider>,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register a provider at a specific tier.
    pub fn register(&mut self, provider: Box<dyn CompletionProvider>, tier: ModelTier) {
        self.providers.push(RegisteredProvider { provider, tier });
    }

    /// Get the provider index for a given recursion depth.
    /// Depth 0 → Tier1, Depth 1 → Tier2, Depth 2+ → Tier3.
    fn tier_for_depth(depth: u8) -> ModelTier {
        match depth {
            0 => ModelTier::Tier1,
            1 => ModelTier::Tier2,
            _ => ModelTier::Tier3,
        }
    }

    /// Select a provider for a given recursion depth.
    /// Falls back to the closest available tier if exact match isn't registered.
    pub fn select_for_depth(&self, depth: u8) -> Option<&dyn CompletionProvider> {
        let target_tier = Self::tier_for_depth(depth);
        // Try exact tier match first
        if let Some(p) = self.providers.iter().find(|p| p.tier == target_tier) {
            return Some(p.provider.as_ref());
        }
        // Fall back: for higher tiers (cheaper), try next lower tier
        // For lower tiers (more capable), try next higher tier
        match target_tier {
            ModelTier::Tier3 => self
                .providers
                .iter()
                .find(|p| p.tier == ModelTier::Tier2)
                .or_else(|| self.providers.iter().find(|p| p.tier == ModelTier::Tier1)),
            ModelTier::Tier2 => self
                .providers
                .iter()
                .find(|p| p.tier == ModelTier::Tier1)
                .or_else(|| self.providers.iter().find(|p| p.tier == ModelTier::Tier3)),
            ModelTier::Tier1 => self
                .providers
                .iter()
                .find(|p| p.tier == ModelTier::Tier2)
                .or_else(|| self.providers.iter().find(|p| p.tier == ModelTier::Tier3)),
        }
        .map(|p| p.provider.as_ref())
    }

    /// Select the cheapest provider that meets the minimum context window requirement.
    pub fn select_cheapest(&self, min_context_tokens: u32) -> Option<&dyn CompletionProvider> {
        self.providers
            .iter()
            .filter(|p| p.provider.capabilities().max_context_tokens >= min_context_tokens)
            .min_by(|a, b| {
                let cost_a = a.provider.capabilities().cost_per_input_token
                    + a.provider.capabilities().cost_per_output_token;
                let cost_b = b.provider.capabilities().cost_per_input_token
                    + b.provider.capabilities().cost_per_output_token;
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.provider.as_ref())
    }

    /// Get the first registered provider (primary).
    pub fn primary(&self) -> Option<&dyn CompletionProvider> {
        self.providers.first().map(|p| p.provider.as_ref())
    }

    /// Get capabilities of the primary provider.
    fn primary_capabilities(&self) -> ProviderCapabilities {
        self.primary()
            .map(|p| p.capabilities())
            .unwrap_or(ProviderCapabilities {
                supports_streaming: false,
                supports_tool_calling: false,
                max_context_tokens: 0,
                cost_per_input_token: 0.0,
                cost_per_output_token: 0.0,
            })
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// A `CompletionProvider` implementation that wraps a `ModelRouter` and adds
/// fallback chain behavior: on retryable errors (429, 5xx), tries the next
/// provider in registration order.
pub struct RouterProvider {
    router: ModelRouter,
}

impl RouterProvider {
    pub fn new(router: ModelRouter) -> Self {
        Self { router }
    }

    /// Access the underlying router for depth-based selection etc.
    pub fn router(&self) -> &ModelRouter {
        &self.router
    }

    fn is_retryable(error: &ProviderError) -> bool {
        match error {
            ProviderError::RateLimited { .. } => true,
            ProviderError::ApiError { status, .. } => {
                matches!(status, 429 | 500 | 502 | 503 | 529)
            }
            ProviderError::RequestFailed(_) => true,
            _ => false,
        }
    }
}

#[async_trait]
impl CompletionProvider for RouterProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let mut last_error = None;

        for (i, rp) in self.router.providers.iter().enumerate() {
            match rp.provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if Self::is_retryable(&e) && i + 1 < self.router.providers.len() {
                        warn!(
                            provider = rp.provider.name(),
                            error = %e,
                            next = self.router.providers[i + 1].provider.name(),
                            "Provider failed, falling back"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ProviderError::Other("No providers registered in router".to_string())
        }))
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        let mut last_error = None;

        for (i, rp) in self.router.providers.iter().enumerate() {
            match rp.provider.complete_streaming(request.clone()).await {
                Ok(stream) => {
                    if i > 0 {
                        info!(
                            provider = rp.provider.name(),
                            "Streaming via fallback provider"
                        );
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    if Self::is_retryable(&e) && i + 1 < self.router.providers.len() {
                        warn!(
                            provider = rp.provider.name(),
                            error = %e,
                            "Stream failed, trying next provider"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ProviderError::Other("No providers registered in router".to_string())
        }))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.router.primary_capabilities()
    }

    fn name(&self) -> &str {
        "router"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use futures::stream;

    struct MockProvider {
        name: &'static str,
        caps: ProviderCapabilities,
        fail: bool,
    }

    #[async_trait]
    impl CompletionProvider for MockProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            if self.fail {
                return Err(ProviderError::ApiError {
                    status: 500,
                    body: "mock failure".to_string(),
                });
            }
            Ok(CompletionResponse {
                message: Message::assistant(format!("from {}", self.name)),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: self.name.to_string(),
            })
        }

        async fn complete_streaming(
            &self,
            _request: CompletionRequest,
        ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
            if self.fail {
                return Err(ProviderError::ApiError {
                    status: 500,
                    body: "mock failure".to_string(),
                });
            }
            Ok(Box::pin(stream::empty()))
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.caps.clone()
        }

        fn name(&self) -> &str {
            self.name
        }
    }

    fn make_caps(cost_in: f64, cost_out: f64, context: u32) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: true,
            supports_tool_calling: true,
            max_context_tokens: context,
            cost_per_input_token: cost_in,
            cost_per_output_token: cost_out,
        }
    }

    #[test]
    fn depth_routing() {
        let mut router = ModelRouter::new();
        router.register(
            Box::new(MockProvider {
                name: "tier1",
                caps: make_caps(0.003, 0.015, 200_000),
                fail: false,
            }),
            ModelTier::Tier1,
        );
        router.register(
            Box::new(MockProvider {
                name: "tier2",
                caps: make_caps(0.001, 0.005, 200_000),
                fail: false,
            }),
            ModelTier::Tier2,
        );
        router.register(
            Box::new(MockProvider {
                name: "tier3",
                caps: make_caps(0.0001, 0.0005, 128_000),
                fail: false,
            }),
            ModelTier::Tier3,
        );

        assert_eq!(router.select_for_depth(0).unwrap().name(), "tier1");
        assert_eq!(router.select_for_depth(1).unwrap().name(), "tier2");
        assert_eq!(router.select_for_depth(2).unwrap().name(), "tier3");
        assert_eq!(router.select_for_depth(5).unwrap().name(), "tier3");
    }

    #[test]
    fn cheapest_selection() {
        let mut router = ModelRouter::new();
        router.register(
            Box::new(MockProvider {
                name: "expensive",
                caps: make_caps(0.01, 0.03, 200_000),
                fail: false,
            }),
            ModelTier::Tier1,
        );
        router.register(
            Box::new(MockProvider {
                name: "cheap",
                caps: make_caps(0.0001, 0.0005, 128_000),
                fail: false,
            }),
            ModelTier::Tier3,
        );

        assert_eq!(router.select_cheapest(0).unwrap().name(), "cheap");
        // With min context requirement that only expensive meets
        assert_eq!(router.select_cheapest(200_000).unwrap().name(), "expensive");
    }

    #[tokio::test]
    async fn fallback_on_failure() {
        let mut router = ModelRouter::new();
        router.register(
            Box::new(MockProvider {
                name: "primary",
                caps: make_caps(0.01, 0.03, 200_000),
                fail: true,
            }),
            ModelTier::Tier1,
        );
        router.register(
            Box::new(MockProvider {
                name: "fallback",
                caps: make_caps(0.001, 0.005, 200_000),
                fail: false,
            }),
            ModelTier::Tier2,
        );

        let rp = RouterProvider::new(router);
        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::user("test")],
            tools: None,
            temperature: None,
            max_tokens: None,
        };

        let response = rp.complete(request).await.unwrap();
        assert_eq!(response.message.content.as_deref(), Some("from fallback"));
    }
}
