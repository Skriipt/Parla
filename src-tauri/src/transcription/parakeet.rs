// Wrapper autour de `parakeet_rs::ParakeetTDT` pour reproduire l'API
// publique de WhisperEngine (load paresseux + transcribe_samples).
//
// Reference VoiceInk : FluidAudio expose `AsrManager.transcribe(samples,
// decoderState)`. Ici on s'appuie sur parakeet-rs qui encapsule deja le
// chargement ONNX, le mel spectrogram, l'encodeur, le joint net et le
// decoder TDT. On expose juste un handle avec rechargement paresseux si
// le repertoire change.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use parakeet_rs::{ParakeetTDT, Transcriber};

struct Loaded {
    path: PathBuf,
    engine: ParakeetTDT,
}

pub struct ParakeetEngine {
    current: Mutex<Option<Loaded>>,
}

impl ParakeetEngine {
    pub fn new() -> Self {
        Self {
            current: Mutex::new(None),
        }
    }

    /// Charge le modele (repertoire contenant config.json + *.onnx + vocab.txt)
    /// si ce n'est pas deja celui-ci qui est charge. Bloquant.
    pub fn ensure_loaded(&self, model_dir: &Path) -> Result<()> {
        let mut guard = self.current.lock();
        if let Some(cur) = guard.as_ref() {
            if cur.path == model_dir {
                return Ok(());
            }
        }
        // Libere l'ancien avant de charger le nouveau.
        *guard = None;

        // Selection de l'execution provider ONNX. parakeet-rs prend un
        // `Option<ExecutionConfig>` ; `None` retombe sur `ExecutionProvider::Cpu`
        // (le `#[default]` de l'enum). Il faut donc lui passer explicitement le
        // provider GPU, sinon compiler avec `cuda-onnx` / `directml-onnx` ne
        // fait qu'exposer le variant sans jamais l'utiliser (inference CPU).
        // Chaque provider GPU retombe automatiquement sur CPU s'il echoue a
        // s'initialiser (parakeet-rs enchaine [GPU, CPU.error_on_failure()]).
        #[cfg(feature = "cuda-onnx")]
        let cfg = Some(
            parakeet_rs::ExecutionConfig::new()
                .with_execution_provider(parakeet_rs::ExecutionProvider::Cuda),
        );
        #[cfg(all(not(feature = "cuda-onnx"), feature = "directml-onnx"))]
        let cfg = Some(
            parakeet_rs::ExecutionConfig::new()
                .with_execution_provider(parakeet_rs::ExecutionProvider::DirectML),
        );
        #[cfg(all(not(feature = "cuda-onnx"), not(feature = "directml-onnx")))]
        let cfg: Option<parakeet_rs::ExecutionConfig> = None;

        let engine = ParakeetTDT::from_pretrained(model_dir, cfg)
            .map_err(|e| anyhow!("parakeet load: {e:?}"))?;
        *guard = Some(Loaded {
            path: model_dir.to_path_buf(),
            engine,
        });
        Ok(())
    }

    /// Transcrit un buffer PCM Float32 mono 16 kHz. Bloquant (inference).
    /// `language` est accepte mais non-utilise : parakeet TDT v2 est anglais
    /// uniquement, v3 detecte la langue automatiquement.
    ///
    /// Au-dela de 15 s, l'audio est decoupe en fenetres comme le fait
    /// FluidAudio (AsrManager+Transcription.swift : `maxModelSamples` puis
    /// `ChunkProcessor`). Une passe unique sur un long audio fait deriver
    /// Parakeet v3 vers l'anglais et lui fait sauter des phrases entieres.
    pub fn transcribe_samples(&self, samples: &[f32], _language: Option<&str>) -> Result<String> {
        let mut guard = self.current.lock();
        let loaded = guard
            .as_mut()
            .ok_or_else(|| anyhow!("modele Parakeet non charge"))?;
        let windows = plan_windows(samples);
        let mut words: Vec<String> = Vec::new();
        for w in &windows {
            // parakeet-rs attend un Vec<f32>. Une copie est inevitable ici.
            let result = loaded
                .engine
                .transcribe_samples(samples[w.start..w.end].to_vec(), SAMPLE_RATE, 1, None)
                .map_err(|e| anyhow!("parakeet transcribe: {e:?}"))?;
            if windows.len() == 1 {
                return Ok(result.text);
            }
            let offset = w.start as f32 / SAMPLE_RATE as f32;
            let keep_from = w.keep_from as f32 / SAMPLE_RATE as f32;
            let keep_to = w.keep_to as f32 / SAMPLE_RATE as f32;
            for (start, text) in group_words(&result.tokens) {
                let t = offset + start;
                if t >= keep_from && t < keep_to {
                    words.push(text);
                }
            }
        }
        Ok(join_words(&words))
    }

    /// Libere la memoire (ONNX session). Utile quand l'utilisateur change de
    /// modele ou desactive la source.
    pub fn unload(&self) {
        *self.current.lock() = None;
    }
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct ParakeetEngineState(pub Arc<ParakeetEngine>);

const SAMPLE_RATE: u32 = 16_000;
/// Plus longue fenetre envoyee en une passe (FluidAudio
/// `ASRConstants.maxModelSamples` = 15 s).
const MAX_WINDOW_SAMPLES: usize = 240_000;
/// Recouvrement entre deux fenetres (FluidAudio
/// `ChunkProcessor.overlapSeconds` = 2 s).
const OVERLAP_SAMPLES: usize = 32_000;
/// Trame d'energie pour chercher le point de coupe le plus silencieux.
const ENERGY_FRAME_SAMPLES: usize = 320;

/// Fenetre `[start, end)` passee au modele. Seuls les mots qui commencent
/// dans `[keep_from, keep_to)` sont gardes : chaque mot vient de la fenetre
/// ou il a le plus de contexte, sans doublon dans le recouvrement.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Window {
    start: usize,
    end: usize,
    keep_from: usize,
    keep_to: usize,
}

fn plan_windows(samples: &[f32]) -> Vec<Window> {
    let n = samples.len();
    if n <= MAX_WINDOW_SAMPLES {
        return vec![Window {
            start: 0,
            end: n,
            keep_from: 0,
            keep_to: n,
        }];
    }
    let stride = MAX_WINDOW_SAMPLES - OVERLAP_SAMPLES;
    let mut starts = vec![0usize];
    while starts[starts.len() - 1] + MAX_WINDOW_SAMPLES < n {
        let last = starts[starts.len() - 1];
        // La derniere fenetre est recalee sur la fin de l'audio pour garder
        // 15 s de contexte au lieu d'une queue tres courte.
        starts.push((last + stride).min(n - MAX_WINDOW_SAMPLES));
    }
    // Coupe entre deux fenetres : point le plus silencieux de la moitie
    // centrale du recouvrement, pour ne pas trancher un mot.
    let cuts: Vec<usize> = starts
        .windows(2)
        .map(|p| {
            let (a, b) = (p[1], p[0] + MAX_WINDOW_SAMPLES);
            let margin = (b - a) / 4;
            quietest_point(samples, a + margin, b - margin)
        })
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| Window {
            start,
            end: (start + MAX_WINDOW_SAMPLES).min(n),
            keep_from: if i == 0 { 0 } else { cuts[i - 1] },
            keep_to: if i == cuts.len() { n } else { cuts[i] },
        })
        .collect()
}

fn quietest_point(samples: &[f32], from: usize, to: usize) -> usize {
    let mut best = (f32::MAX, (from + to) / 2);
    let mut pos = from;
    while pos + ENERGY_FRAME_SAMPLES <= to {
        let energy: f32 = samples[pos..pos + ENERGY_FRAME_SAMPLES]
            .iter()
            .map(|s| s * s)
            .sum();
        if energy < best.0 {
            best = (energy, pos + ENERGY_FRAME_SAMPLES / 2);
        }
        pos += ENERGY_FRAME_SAMPLES;
    }
    best.1
}

/// Regroupe les tokens SentencePiece en mots (un token qui commence par un
/// espace ouvre un mot) avec l'instant de debut du mot, relatif a la fenetre.
fn group_words(tokens: &[parakeet_rs::TimedToken]) -> Vec<(f32, String)> {
    let mut words: Vec<(f32, String)> = Vec::new();
    for t in tokens {
        match words.last_mut() {
            Some((_, text)) if !t.text.starts_with(char::is_whitespace) => text.push_str(&t.text),
            _ => words.push((t.start, t.text.clone())),
        }
    }
    words
}

fn join_words(words: &[String]) -> String {
    let mut out = String::new();
    for w in words {
        if !out.is_empty() && !w.starts_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push_str(w);
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_is_a_single_window() {
        let samples = vec![0.0; MAX_WINDOW_SAMPLES];
        let w = plan_windows(&samples);
        assert_eq!(w.len(), 1);
        assert_eq!((w[0].start, w[0].end), (0, MAX_WINDOW_SAMPLES));
    }

    #[test]
    fn long_audio_windows_cover_everything_once() {
        // 147 s, la duree de l'enregistrement qui a revele le bug.
        let n = 147 * 16_000 + 1_234;
        let samples: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w = plan_windows(&samples);
        assert!(w.len() > 1);
        assert_eq!(w[0].keep_from, 0);
        assert_eq!(w[w.len() - 1].keep_to, n);
        assert_eq!(w[w.len() - 1].end, n);
        for win in &w {
            assert!(win.end - win.start <= MAX_WINDOW_SAMPLES);
            assert!(win.start <= win.keep_from && win.keep_to <= win.end);
        }
        for pair in w.windows(2) {
            assert_eq!(pair[0].keep_to, pair[1].keep_from);
        }
    }

    #[test]
    fn cut_lands_on_silence() {
        let n = MAX_WINDOW_SAMPLES + 20_000;
        let mut samples = vec![0.5f32; n];
        let silence = 130_000..130_640;
        for s in &mut samples[silence.clone()] {
            *s = 0.0;
        }
        let w = plan_windows(&samples);
        assert_eq!(w.len(), 2);
        assert!(silence.contains(&w[0].keep_to));
    }

    #[test]
    fn words_regroup_subword_tokens() {
        let tok = |text: &str, start: f32| parakeet_rs::TimedToken {
            text: text.to_string(),
            start,
            end: start + 0.1,
        };
        let words = group_words(&[
            tok(" l'en", 1.0),
            tok("ti", 1.1),
            tok("èreté", 1.2),
            tok(",", 1.3),
            tok(" du", 1.5),
        ]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], (1.0, " l'entièreté,".to_string()));
        assert_eq!(join_words(&[words[0].1.clone(), "texte".into()]), "l'entièreté, texte");
    }

    /// Compare passe unique et fenetres sur un vrai enregistrement.
    /// PARLA_PARAKEET_DIR=<modele> PARLA_WAV=<fichier.wav>
    /// cargo test --lib parakeet_long_audio -- --ignored --nocapture
    #[test]
    #[ignore]
    fn parakeet_long_audio() {
        let dir = std::env::var("PARLA_PARAKEET_DIR").expect("PARLA_PARAKEET_DIR");
        let wav = std::env::var("PARLA_WAV").expect("PARLA_WAV");
        let samples = crate::transcription::whisper::read_wav_as_f32(Path::new(&wav)).unwrap();
        let engine = ParakeetEngine::new();
        engine.ensure_loaded(Path::new(&dir)).unwrap();
        let single = {
            let mut g = engine.current.lock();
            g.as_mut()
                .unwrap()
                .engine
                .transcribe_samples(samples.clone(), SAMPLE_RATE, 1, None)
                .unwrap()
                .text
        };
        println!("=== passe unique ({} car.)\n{single}\n", single.len());
        let chunked = engine.transcribe_samples(&samples, None).unwrap();
        println!("=== fenetres ({} car.)\n{chunked}\n", chunked.len());
    }
}
