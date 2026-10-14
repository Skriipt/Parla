// xAI Grok speech-to-text en batch.
//
// Reference VoiceInk : LLMkit XAIClient.swift.
// Endpoint : POST https://api.x.ai/v1/stt
// Auth     : Authorization: Bearer <apiKey>
// Body     : multipart/form-data, le champ `file` doit etre LE DERNIER
//            (contrainte xAI). `language` optionnel, "auto" = on n'envoie
//            rien. `format=true` active l'inverse text normalization (ITN)
//            uniquement quand une langue est explicite.
// Reponse  : JSON `{"text": "..."}`.
// Verify   : GET /v1/api-key avec Bearer ; 200 = OK.

use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::multipart::Form;
use serde::Deserialize;

use super::http::{batch_client, map_http_err, wav_part_from_path};
use super::provider::{CloudTranscriptionProvider, TranscribeRequest};

pub struct XaiProvider;

#[derive(Debug, Deserialize)]
struct XaiResponse {
    text: Option<String>,
}

#[async_trait]
impl CloudTranscriptionProvider for XaiProvider {
    fn id(&self) -> &'static str {
        "xai"
    }

    async fn verify_api_key(&self, api_key: &str) -> Result<()> {
        let client = batch_client()?;
        let resp = client
            .get("https://api.x.ai/v1/api-key")
            .bearer_auth(api_key)
            .send()
            .await
            .map_err(map_http_err)?;
        if !resp.status().is_success() {
            anyhow::bail!("HTTP {} (cle API invalide ?)", resp.status());
        }
        Ok(())
    }

    async fn transcribe(
        &self,
        wav_path: &Path,
        api_key: &str,
        request: &TranscribeRequest,
    ) -> Result<String> {
        // Ordre des champs important pour xAI : `language` et `format` AVANT
        // `file`, ce dernier doit etre en dernier (cf. LLMkit XAIClient L37).
        let mut form = Form::new();

        let lang_provided = match request.language.as_deref() {
            Some(lang) if !lang.is_empty() && lang != "auto" => {
                form = form.text("language", lang.to_string());
                true
            }
            _ => false,
        };
        if lang_provided {
            // ITN n'a de sens que quand on connait la langue.
            form = form.text("format", "true");
        }

        // Pas de champ `model` : xAI choisit son modele STT par defaut
        // (Grok Voice Transcribe 2.0), comme LLMkit XAIClient (37100b2).
        // Vocabulaire personnalise : un champ `keyterm` par terme.
        for term in keyterms(&request.custom_vocabulary) {
            form = form.text("keyterm", term);
        }

        form = form.part("file", wav_part_from_path(wav_path).await?);

        let client = batch_client()?;
        let resp = client
            .post("https://api.x.ai/v1/stt")
            .bearer_auth(api_key)
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .map_err(map_http_err)?;

        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            let msg = String::from_utf8_lossy(&body);
            anyhow::bail!("HTTP {status}: {msg}");
        }

        match serde_json::from_slice::<XaiResponse>(&body) {
            Ok(r) => Ok(r.text.unwrap_or_default()),
            Err(_) => Ok(String::from_utf8_lossy(&body).trim().to_string()),
        }
    }
}

/// LLMkit XAIClient / XAIStreamingClient : termes trimmes, non vides,
/// 50 caracteres max, 100 max.
pub(crate) fn keyterms(raw: &[String]) -> Vec<String> {
    raw.iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty() && t.chars().count() <= 50)
        .take(100)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod keyterm_tests {
    use super::keyterms;

    #[test]
    fn keyterms_trim_filter_and_cap() {
        let mut raw: Vec<String> = vec![" Parla ".into(), "".into(), "y".repeat(51)];
        raw.extend((0..150).map(|i| format!("t{i}")));
        let k = keyterms(&raw);
        assert_eq!(k.first().map(String::as_str), Some("Parla"));
        assert_eq!(k.len(), 100);
        assert!(k.iter().all(|t| t.chars().count() <= 50));
    }
}
