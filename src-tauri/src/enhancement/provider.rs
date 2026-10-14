// Trait abstrait pour un provider LLM d'enhancement.
//
// Reference VoiceInk : VoiceInk/Services/AIEnhancement/AIService.swift.
// VoiceInk delegue a LLMkit (AnthropicLLMClient, OpenAILLMClient, OllamaClient)
// qui exposent un chatCompletion non-streaming. On reproduit la meme sortie.

use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

/// Parametres de raisonnement, propres a certains modeles (gpt-5*, gemini 2.5*,
/// groq gpt-oss, cerebras). Refere au code VoiceInk ReasoningConfig
/// (AIService.swift L~200-260) pour les valeurs par modele.
#[derive(Debug, Clone, Default)]
pub struct ReasoningConfig {
    /// Optionnel : "none" | "minimal" | "low" | "medium" | "high".
    pub effort: Option<String>,
    /// Champs supplementaires injectes tel quel dans le body JSON.
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct EnhancementRequest {
    /// Systeme : prompt + contextes ensemble (deja assemble par le service).
    pub system_prompt: String,
    /// Message utilisateur (deja wrappe <TRANSCRIPT>...</TRANSCRIPT>).
    pub user_message: String,
    /// Modele choisi (ex: "claude-sonnet-4-6", "gpt-4.1", "mistral-large-latest").
    pub model: String,
    /// Temperature : 1.0 pour gpt-5*, 0.3 sinon (VoiceInk AIService L~140).
    pub temperature: f32,
    /// Reasoning config optionnelle.
    pub reasoning: ReasoningConfig,
    /// Timeout requete (VoiceInk defaut 7s).
    pub timeout: Duration,
    /// Override optionnel de l'endpoint (utilise par Ollama et Custom
    /// qui ont des URLs configurees par l'utilisateur).
    pub endpoint_override: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EnhancementResponse {
    pub text: String,
}

/// Trait commun a tous les providers LLM d'enhancement.
/// Non-streaming (VoiceInk ne streame pas l'enhancement).
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Identifiant stable utilise en store et keyring (match VoiceInk AIProvider.rawValue).
    fn id(&self) -> &'static str;

    /// Libelle pour l'UI (ex "OpenAI", "Groq", "Cerebras").
    fn label(&self) -> &'static str;

    /// Liste des modeles proposes par defaut. Liste pouvant etre vide pour
    /// les providers dynamiques (Ollama, OpenRouter) qui interrogent leur
    /// serveur.
    fn default_models(&self) -> &'static [&'static str];

    /// Modele par defaut (premier choix). VoiceInk AIService L~70-130.
    fn default_model(&self) -> &'static str;

    /// Endpoint utilise (documentation / UI).
    fn endpoint(&self) -> &'static str;

    /// Indique si ce provider requiert une cle API.
    fn requires_api_key(&self) -> bool {
        true
    }

    /// Applique-t-on le rate limit global de 1s ? VoiceInk n'applique pas
    /// le rate limit aux providers locaux (Ollama, LocalCLI).
    fn rate_limited(&self) -> bool {
        true
    }

    /// Le provider accepte-t-il un identifiant de modele libre en plus de
    /// la liste proposee ? Reference VoiceInk AIProvider.supportsCustomModelID
    /// (cerebras, groq, gemini, anthropic, openAI, mistral). Parla l'active
    /// aussi pour OpenRouter, dont VoiceInk charge le catalogue dynamiquement
    /// alors que Parla n'en propose qu'une courte liste.
    fn supports_custom_model_id(&self) -> bool {
        false
    }

    /// Effectue la requete d'enhancement et renvoie le texte brut de reponse.
    /// Le filtre <thinking>/<think>/<reasoning> est applique par le service.
    async fn chat_completion(
        &self,
        api_key: &str,
        req: &EnhancementRequest,
    ) -> Result<EnhancementResponse>;
}
