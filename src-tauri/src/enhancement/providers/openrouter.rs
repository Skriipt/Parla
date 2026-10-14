// OpenRouter chat completions.
//
// Reference VoiceInk : AIService.swift (case .openRouter) - baseURL
// https://openrouter.ai/api/v1/chat/completions.
// La liste de modeles est dynamique, fetched depuis /models cote VoiceInk.
// Ici on expose un modele par defaut et on laisse le user entrer librement.

use anyhow::Result;
use async_trait::async_trait;

use crate::enhancement::provider::{EnhancementRequest, EnhancementResponse, LLMProvider};

use super::openai_compat;

pub struct OpenRouterProvider;

// Courte liste de suggestions ; VoiceInk 2.22 charge le catalogue OpenRouter
// dynamiquement avec pour defaut qwen/qwen3.8-27b. Parla accepte un
// identifiant libre a la place.
const MODELS: &[&str] = &[
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "anthropic/claude-sonnet-4-6",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct",
];

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn id(&self) -> &'static str {
        "openrouter"
    }
    fn label(&self) -> &'static str {
        "OpenRouter"
    }
    fn default_models(&self) -> &'static [&'static str] {
        MODELS
    }
    fn default_model(&self) -> &'static str {
        "qwen/qwen3.8-27b"
    }
    /// VoiceInk AIProvider.supportsCustomModelID (commit 7929145d) : la
    /// liste ne fait que proposer des modeles, l'utilisateur peut saisir
    /// n'importe quel identifiant accepte par le provider.
    fn supports_custom_model_id(&self) -> bool {
        true
    }
    fn endpoint(&self) -> &'static str {
        "https://openrouter.ai/api/v1/chat/completions"
    }

    async fn chat_completion(
        &self,
        api_key: &str,
        req: &EnhancementRequest,
    ) -> Result<EnhancementResponse> {
        openai_compat::chat_completion(self.endpoint(), api_key, req).await
    }
}
