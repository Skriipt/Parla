// ElevenLabs Scribe v1/v2 en batch.
//
// Reference VoiceInk : LLMkit ElevenLabsClient.swift.
// Endpoint : POST https://api.elevenlabs.io/v1/speech-to-text
// Headers : xi-api-key: <key> (pas Bearer)
// Body multipart : file, model_id, temperature=0.0, tag_audio_events=false,
// language_code? (si langue non vide).

use std::path::Path;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::multipart::Form;
use serde::Deserialize;

use super::http::{batch_client, map_http_err, wav_part_from_path};
use super::provider::{CloudTranscriptionProvider, TranscribeRequest};

pub struct ElevenLabsProvider;

#[derive(Debug, Deserialize)]
struct ElevenLabsResponse {
    text: String,
}

#[async_trait]
impl CloudTranscriptionProvider for ElevenLabsProvider {
    fn id(&self) -> &'static str {
        "elevenlabs"
    }

    async fn verify_api_key(&self, api_key: &str) -> Result<()> {
        let client = batch_client()?;
        let resp = client
            .get("https://api.elevenlabs.io/v1/user")
            .header("xi-api-key", api_key)
            .send()
            .await
            .map_err(map_http_err)?;
        if !resp.status().is_success() {
            anyhow::bail!("HTTP {}", resp.status());
        }
        Ok(())
    }

    async fn transcribe(
        &self,
        wav_path: &Path,
        api_key: &str,
        request: &TranscribeRequest,
    ) -> Result<String> {
        let mut form = Form::new()
            .part("file", wav_part_from_path(wav_path).await?)
            .text("model_id", request.model.clone())
            .text("temperature", "0.0")
            .text("tag_audio_events", "false");

        if let Some(lang) = request.language.as_deref() {
            if !lang.is_empty() && lang != "auto" {
                form = form.text("language_code", lang.to_string());
            }
        }

        // LLMkit ElevenLabsClient (VoiceInk 2.22) : hesitations retirees et
        // vocabulaire personnalise envoye comme `keyterms`.
        form = form.text("no_verbatim", "true");
        for term in normalize_keyterms(&request.custom_vocabulary) {
            form = form.text("keyterms", term);
        }

        let client = batch_client()?;
        let resp = client
            .post("https://api.elevenlabs.io/v1/speech-to-text")
            .header("xi-api-key", api_key)
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .map_err(map_http_err)?;

        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            anyhow::bail!("HTTP {status}: {}", String::from_utf8_lossy(&body));
        }

        let parsed: ElevenLabsResponse = serde_json::from_slice(&body)
            .map_err(|e| anyhow!("parse JSON: {e}"))?;
        Ok(parsed.text)
    }
}

/// LLMkit ElevenLabsClient.normalizedKeyterms : trim, 50 caracteres et
/// 5 mots max, sans < > { } [ ] \\, dedup insensible a la casse, 1000 max.
fn normalize_keyterms(raw: &[String]) -> Vec<String> {
    const UNSUPPORTED: [char; 7] = ['<', '>', '{', '}', '[', ']', '\\'];
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for term in raw {
        let trimmed = term.trim();
        if trimmed.is_empty()
            || trimmed.chars().count() > 50
            || trimmed.split_whitespace().count() > 5
            || trimmed.chars().any(|c| UNSUPPORTED.contains(&c))
        {
            continue;
        }
        if !seen.insert(trimmed.to_lowercase()) {
            continue;
        }
        out.push(trimmed.to_string());
        if out.len() == 1_000 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod keyterm_tests {
    use super::normalize_keyterms;

    #[test]
    fn batch_keyterms_follow_llmkit_rules() {
        let raw: Vec<String> = vec![
            " Docker ".into(),
            "DOCKER".into(),
            "one two three four five six".into(),
            "a{b".into(),
            "x".repeat(51),
            "Kubernetes cluster".into(),
        ];
        assert_eq!(
            normalize_keyterms(&raw),
            vec!["Docker".to_string(), "Kubernetes cluster".to_string()]
        );
    }
}
