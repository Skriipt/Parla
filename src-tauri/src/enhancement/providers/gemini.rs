// Gemini via shim OpenAI.
//
// Reference VoiceInk : AIService.swift (case .gemini) - baseURL
// https://generativelanguage.googleapis.com/v1beta/openai/chat/completions.

use anyhow::Result;
use async_trait::async_trait;

use crate::enhancement::provider::{EnhancementRequest, EnhancementResponse, LLMProvider};

use super::openai_compat;

pub struct GeminiProvider;

// VoiceInk 2.22 AIService.availableModels (.gemini), commit b890763a
// "Add Gemini 3.8 Flash as the default enhancement model".
const MODELS: &[&str] = &[
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
];

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn id(&self) -> &'static str {
        "gemini"
    }
    fn label(&self) -> &'static str {
        "Google Gemini"
    }
    fn default_models(&self) -> &'static [&'static str] {
        MODELS
    }
    fn default_model(&self) -> &'static str {
        "gemini-3.8-flash"
    }
    /// VoiceInk AIProvider.supportsCustomModelID (commit 7929145d) : la
    /// liste ne fait que proposer des modeles, l'utilisateur peut saisir
    /// n'importe quel identifiant accepte par le provider.
    fn supports_custom_model_id(&self) -> bool {
        true
    }
    fn endpoint(&self) -> &'static str {
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    }

    async fn chat_completion(
        &self,
        api_key: &str,
        req: &EnhancementRequest,
    ) -> Result<EnhancementResponse> {
        openai_compat::chat_completion(self.endpoint(), api_key, req).await
    }
}
