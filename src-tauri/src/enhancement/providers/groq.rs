// Groq chat completions.
//
// Reference VoiceInk : AIService.swift (case .groq) - baseURL
// https://api.groq.com/openai/v1/chat/completions.

use anyhow::Result;
use async_trait::async_trait;

use crate::enhancement::provider::{EnhancementRequest, EnhancementResponse, LLMProvider};

use super::openai_compat;

pub struct GroqProvider;

// VoiceInk 2.22 AIService.availableModels (.groq), commit 246599e0.
const MODELS: &[&str] = &[
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
];

#[async_trait]
impl LLMProvider for GroqProvider {
    fn id(&self) -> &'static str {
        "groq"
    }
    fn label(&self) -> &'static str {
        "Groq"
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
        "https://api.groq.com/openai/v1/chat/completions"
    }

    async fn chat_completion(
        &self,
        api_key: &str,
        req: &EnhancementRequest,
    ) -> Result<EnhancementResponse> {
        openai_compat::chat_completion(self.endpoint(), api_key, req).await
    }
}
