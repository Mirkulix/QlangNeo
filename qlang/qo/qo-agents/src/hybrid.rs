//! Hybrid Organism — 1 Core-LLM + N kleine Spezialisten über QLANG-Graph.
//!
//! Flow pro Anfrage:
//!
//! ```text
//!                     ┌──────────── Specialist A ──► fast answer
//! input → score ─► max├──────────── Specialist B
//!                     │             Specialist ...
//!                     └─if score<τ─► Core-LLM (Op::OllamaChat via Graph)
//! ```
//!
//! - Spezialisten sind `Send + Sync` und synchron (lokale zero-multiply Inferenz).
//! - Der Core-Call baut pro Request einen QLANG-Graph mit `Op::OllamaChat`
//!   und läuft durch den Standard-Executor → QLMS-signierbar, inspektierbar.
//! - Metriken (Counter pro Tier, avg Latenz) sind atomar → thread-safe.
//!
//! Keine externe Abhängigkeit außer dem bereits vorhandenen qlang-* Stack.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Eine Spezialist-Komponente im Hybrid-Organism.
///
/// Implementationen MÜSSEN `score` billig halten (< 1 ms), denn jede Anfrage
/// ruft es auf allen Spezialisten auf.
pub trait Specialist: Send + Sync {
    /// Eindeutiger Name (für Logs, Metriken, QLMS-Graph-Knoten).
    fn name(&self) -> &str;

    /// Konfidenz in `[0.0, 1.0]`, dass dieser Spezialist die Anfrage beantworten kann.
    fn score(&self, input: &str) -> f32;

    /// Erzeugt die Antwort. Läuft synchron (lokale Inferenz); bei Bedarf
    /// intern parallelisierbar.
    fn handle(&self, input: &str) -> String;

    /// Geschätzte durchschnittliche Latenz in ms (Info, nicht Hard-Budget).
    fn avg_latency_ms(&self) -> u64 {
        5
    }
}

/// Konfiguration für den Hybrid-Organism.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Ollama-Modell für den Core-LLM (z. B. `gemma3:27b`, `qwen2.5:7b`).
    pub core_model: String,
    /// Mindest-Score, damit ein Spezialist genommen wird. Darunter → Core-LLM.
    /// Üblicher Startwert: `0.7`.
    pub specialist_threshold: f32,
    /// Soll Core-LLM als Fallback dienen? Wenn `false` und kein Spezialist
    /// überschreitet die Schwelle, wird `HybridError::NoHandler` zurückgegeben.
    pub enable_core_fallback: bool,
    /// Optionaler System-Prompt, der beim Core-LLM-Call vorangestellt wird.
    pub core_system_prompt: Option<String>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        let core_model = std::env::var("HYBRID_CORE_MODEL")
            .unwrap_or_else(|_| "qwen2.5:3b".to_string());
        let threshold = std::env::var("HYBRID_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.7)
            .clamp(0.0, 1.0);
        Self {
            core_model,
            specialist_threshold: threshold,
            enable_core_fallback: true,
            core_system_prompt: None,
        }
    }
}

/// Welcher Tier hat die Antwort geliefert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Specialist,
    Core,
}

/// Antwort des Organism.
#[derive(Debug, Clone)]
pub struct HybridResponse {
    pub text: String,
    pub handler: String,
    pub tier: Tier,
    pub confidence: f32,
    pub latency_ms: u64,
    /// Wie viele Nodes im QLANG-Graph ausgeführt wurden (0 bei Spezialist-Pfad).
    pub graph_nodes: usize,
}

/// Fehlertypen.
#[derive(Debug, thiserror::Error)]
pub enum HybridError {
    #[error("no specialist exceeded threshold and core fallback is disabled")]
    NoHandler,
    #[error("core LLM graph execution failed: {0}")]
    CoreExecution(String),
}

/// Atomar gezählte Metriken.
#[derive(Debug, Default)]
pub struct HybridStats {
    pub specialist_hits: AtomicU64,
    pub core_hits: AtomicU64,
    pub total_specialist_latency_ms: AtomicU64,
    pub total_core_latency_ms: AtomicU64,
    pub per_specialist_hits: std::sync::Mutex<HashMap<String, u64>>,
}

impl HybridStats {
    pub fn snapshot(&self) -> StatsSnapshot {
        let sp = self.specialist_hits.load(Ordering::Relaxed);
        let co = self.core_hits.load(Ordering::Relaxed);
        let sp_ms = self.total_specialist_latency_ms.load(Ordering::Relaxed);
        let co_ms = self.total_core_latency_ms.load(Ordering::Relaxed);
        StatsSnapshot {
            specialist_hits: sp,
            core_hits: co,
            avg_specialist_latency_ms: if sp > 0 { sp_ms / sp } else { 0 },
            avg_core_latency_ms: if co > 0 { co_ms / co } else { 0 },
            per_specialist: self.per_specialist_hits.lock().ok()
                .map(|g| g.clone()).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StatsSnapshot {
    pub specialist_hits: u64,
    pub core_hits: u64,
    pub avg_specialist_latency_ms: u64,
    pub avg_core_latency_ms: u64,
    pub per_specialist: HashMap<String, u64>,
}

/// Der Hybrid-Organism.
pub struct HybridOrganism {
    config: HybridConfig,
    specialists: Vec<Arc<dyn Specialist>>,
    stats: Arc<HybridStats>,
}

impl HybridOrganism {
    pub fn new(config: HybridConfig) -> Self {
        Self {
            config,
            specialists: Vec::new(),
            stats: Arc::new(HybridStats::default()),
        }
    }

    /// Registriert einen Spezialisten. Reihenfolge ist irrelevant — gewinnt
    /// wird immer der mit höchstem Score.
    pub fn add_specialist<S: Specialist + 'static>(&mut self, s: S) -> &mut Self {
        self.specialists.push(Arc::new(s));
        self
    }

    pub fn stats(&self) -> Arc<HybridStats> {
        self.stats.clone()
    }

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn specialist_names(&self) -> Vec<String> {
        self.specialists.iter().map(|s| s.name().to_string()).collect()
    }

    /// Pick: höchster Score. Bei Gleichstand gewinnt der erste registrierte
    /// (deterministisch).
    fn pick_best(&self, input: &str) -> Option<(Arc<dyn Specialist>, f32)> {
        let mut best: Option<(Arc<dyn Specialist>, f32)> = None;
        for spec in &self.specialists {
            let s = spec.score(input).clamp(0.0, 1.0);
            match &best {
                Some((_, bs)) if *bs >= s => {}
                _ => best = Some((spec.clone(), s)),
            }
        }
        best
    }

    /// Hauptdispatch. Führt die Anfrage entweder lokal über einen Spezialisten
    /// aus oder — wenn niemand die Schwelle überschreitet — über einen
    /// QLANG-Graph mit `Op::OllamaChat`-Knoten auf dem Core-LLM.
    pub async fn query(&self, input: &str) -> Result<HybridResponse, HybridError> {
        // 1) Klein-Spezialisten befragen (billig).
        let best = self.pick_best(input);

        if let Some((spec, score)) = &best {
            if *score >= self.config.specialist_threshold {
                let start = Instant::now();
                let text = spec.handle(input);
                let latency = start.elapsed().as_millis() as u64;

                self.stats.specialist_hits.fetch_add(1, Ordering::Relaxed);
                self.stats.total_specialist_latency_ms
                    .fetch_add(latency, Ordering::Relaxed);
                if let Ok(mut m) = self.stats.per_specialist_hits.lock() {
                    *m.entry(spec.name().to_string()).or_insert(0) += 1;
                }

                return Ok(HybridResponse {
                    text,
                    handler: spec.name().to_string(),
                    tier: Tier::Specialist,
                    confidence: *score,
                    latency_ms: latency,
                    graph_nodes: 0,
                });
            }
        }

        // 2) Fallback auf Core-LLM.
        if !self.config.enable_core_fallback {
            return Err(HybridError::NoHandler);
        }

        let model = self.config.core_model.clone();
        let system_prompt = self.config.core_system_prompt.clone();
        let input_owned = input.to_string();

        let start = Instant::now();
        let (text, nodes) = tokio::task::spawn_blocking(move || {
            execute_core_llm_graph(&model, system_prompt.as_deref(), &input_owned)
        })
        .await
        .map_err(|e| HybridError::CoreExecution(format!("join error: {e}")))??;
        let latency = start.elapsed().as_millis() as u64;

        self.stats.core_hits.fetch_add(1, Ordering::Relaxed);
        self.stats.total_core_latency_ms
            .fetch_add(latency, Ordering::Relaxed);

        let confidence = best.map(|(_, s)| s).unwrap_or(0.0);
        Ok(HybridResponse {
            text,
            handler: format!("core:{}", self.config.core_model),
            tier: Tier::Core,
            confidence,
            latency_ms: latency,
            graph_nodes: nodes,
        })
    }
}

// ---------------------------------------------------------------------------
// Core-LLM-Pfad: QLANG-Graph mit OllamaChat-Knoten
// ---------------------------------------------------------------------------

fn execute_core_llm_graph(
    model: &str,
    system_prompt: Option<&str>,
    user_input: &str,
) -> Result<(String, usize), HybridError> {
    let mut g = Graph::new(format!("hybrid_core_{}", model.replace([':', '/'], "_")));
    let str_ty = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input_node = g.add_node(
        Op::Input { name: "prompt".into() },
        vec![],
        vec![str_ty.clone()],
    );
    let chat_node = g.add_node(
        Op::OllamaChat { model: model.to_string() },
        vec![str_ty.clone()],
        vec![str_ty.clone()],
    );
    let output_node = g.add_node(
        Op::Output { name: "answer".into() },
        vec![str_ty.clone()],
        vec![],
    );

    g.add_edge(input_node, 0, chat_node, 0, str_ty.clone());
    g.add_edge(chat_node, 0, output_node, 0, str_ty.clone());

    let messages = match system_prompt {
        Some(sys) => serde_json::json!([
            {"role": "system", "content": sys},
            {"role": "user",   "content": user_input},
        ]),
        None => serde_json::json!([
            {"role": "user", "content": user_input},
        ]),
    };

    let mut inputs = HashMap::new();
    inputs.insert(
        "prompt".to_string(),
        TensorData::from_string(&messages.to_string()),
    );

    let result = qlang_runtime::executor::execute(&g, inputs)
        .map_err(|e| HybridError::CoreExecution(e.to_string()))?;

    let text = result
        .outputs
        .get("answer")
        .and_then(|t| t.as_string())
        .unwrap_or_else(|| "<empty core response>".to_string());

    Ok((text, result.stats.nodes_executed))
}

// ---------------------------------------------------------------------------
// Zwei fertige Spezialist-Bausteine (minimal, nützlich als Startpunkt)
// ---------------------------------------------------------------------------

/// Keyword-Spezialist: löst aus, wenn genug Trigger-Wörter matchen.
///
/// Score = `matches / triggers.len()`, 0 wenn keine Trigger.
pub struct KeywordSpecialist {
    name: String,
    triggers: Vec<String>,
    response_fn: Box<dyn Fn(&str) -> String + Send + Sync>,
}

impl KeywordSpecialist {
    pub fn new<N, F>(name: N, triggers: Vec<&str>, response: F) -> Self
    where
        N: Into<String>,
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            triggers: triggers.into_iter().map(|s| s.to_lowercase()).collect(),
            response_fn: Box::new(response),
        }
    }
}

impl Specialist for KeywordSpecialist {
    fn name(&self) -> &str {
        &self.name
    }

    fn score(&self, input: &str) -> f32 {
        if self.triggers.is_empty() {
            return 0.0;
        }
        let lower = input.to_lowercase();
        let hits = self.triggers.iter().filter(|t| lower.contains(t.as_str())).count();
        hits as f32 / self.triggers.len() as f32
    }

    fn handle(&self, input: &str) -> String {
        (self.response_fn)(input)
    }
}

/// Closure-Spezialist: vollständig frei definierte Score- und Handle-Funktion.
pub struct ClosureSpecialist {
    name: String,
    score_fn: Box<dyn Fn(&str) -> f32 + Send + Sync>,
    handle_fn: Box<dyn Fn(&str) -> String + Send + Sync>,
    avg_latency_ms: u64,
}

impl ClosureSpecialist {
    pub fn new<N, S, H>(name: N, avg_latency_ms: u64, score: S, handle: H) -> Self
    where
        N: Into<String>,
        S: Fn(&str) -> f32 + Send + Sync + 'static,
        H: Fn(&str) -> String + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            score_fn: Box::new(score),
            handle_fn: Box::new(handle),
            avg_latency_ms,
        }
    }
}

impl Specialist for ClosureSpecialist {
    fn name(&self) -> &str {
        &self.name
    }
    fn score(&self, input: &str) -> f32 {
        (self.score_fn)(input).clamp(0.0, 1.0)
    }
    fn handle(&self, input: &str) -> String {
        (self.handle_fn)(input)
    }
    fn avg_latency_ms(&self) -> u64 {
        self.avg_latency_ms
    }
}

// ---------------------------------------------------------------------------
// Tests — laufen offline, Core-LLM-Pfad wird mit deaktiviertem Fallback geprüft
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn greet_specialist() -> KeywordSpecialist {
        KeywordSpecialist::new(
            "greeter",
            vec!["hi", "hallo", "hello"],
            |_| "Hi there!".to_string(),
        )
    }

    fn math_specialist() -> ClosureSpecialist {
        ClosureSpecialist::new(
            "math",
            2,
            |i| {
                let has_digit = i.chars().any(|c| c.is_ascii_digit());
                let has_op = i.contains(['+', '-', '*', '/']);
                if has_digit && has_op { 0.95 } else { 0.0 }
            },
            |i| format!("math-spec got: {}", i),
        )
    }

    #[test]
    fn pick_best_returns_highest_score() {
        let mut org = HybridOrganism::new(HybridConfig::default());
        org.add_specialist(greet_specialist())
            .add_specialist(math_specialist());
        let (spec, score) = org.pick_best("hallo world").unwrap();
        assert_eq!(spec.name(), "greeter");
        assert!(score > 0.0);
    }

    #[test]
    fn pick_best_deterministic_on_tie() {
        let mut org = HybridOrganism::new(HybridConfig::default());
        let a = ClosureSpecialist::new("a", 1, |_| 0.5, |_| "A".into());
        let b = ClosureSpecialist::new("b", 1, |_| 0.5, |_| "B".into());
        org.add_specialist(a).add_specialist(b);
        let (spec, _) = org.pick_best("anything").unwrap();
        assert_eq!(spec.name(), "a");
    }

    #[test]
    fn keyword_score_is_fraction_of_triggers() {
        let s = KeywordSpecialist::new("t", vec!["foo", "bar"], |_| "ok".into());
        assert_eq!(s.score("foo bar baz"), 1.0);
        assert_eq!(s.score("foo only"), 0.5);
        assert_eq!(s.score("nope"), 0.0);
    }

    #[tokio::test]
    async fn specialist_path_fires_when_above_threshold() {
        let mut cfg = HybridConfig::default();
        cfg.specialist_threshold = 0.3;
        cfg.enable_core_fallback = false; // erzwingt Spezialist-Pfad
        let mut org = HybridOrganism::new(cfg);
        org.add_specialist(greet_specialist());

        let resp = org.query("hallo welt").await.unwrap();
        assert_eq!(resp.tier, Tier::Specialist);
        assert_eq!(resp.handler, "greeter");
        assert_eq!(resp.graph_nodes, 0);
        assert!(resp.confidence > 0.0);

        let snap = org.stats().snapshot();
        assert_eq!(snap.specialist_hits, 1);
        assert_eq!(snap.core_hits, 0);
        assert_eq!(snap.per_specialist.get("greeter").copied(), Some(1));
    }

    #[tokio::test]
    async fn core_fallback_disabled_yields_error() {
        let mut cfg = HybridConfig::default();
        cfg.specialist_threshold = 0.99;
        cfg.enable_core_fallback = false;
        let mut org = HybridOrganism::new(cfg);
        org.add_specialist(greet_specialist()); // Score 1/3 bei "hi" → < 0.99

        let err = org.query("hi").await.unwrap_err();
        matches!(err, HybridError::NoHandler);
    }

    #[test]
    fn default_config_reads_env() {
        std::env::set_var("HYBRID_CORE_MODEL", "gemma3:27b");
        std::env::set_var("HYBRID_THRESHOLD", "0.42");
        let cfg = HybridConfig::default();
        assert_eq!(cfg.core_model, "gemma3:27b");
        assert!((cfg.specialist_threshold - 0.42).abs() < 1e-6);
        std::env::remove_var("HYBRID_CORE_MODEL");
        std::env::remove_var("HYBRID_THRESHOLD");
    }

    #[test]
    fn threshold_is_clamped() {
        std::env::set_var("HYBRID_THRESHOLD", "5.0");
        let cfg = HybridConfig::default();
        assert_eq!(cfg.specialist_threshold, 1.0);
        std::env::set_var("HYBRID_THRESHOLD", "-0.5");
        let cfg = HybridConfig::default();
        assert_eq!(cfg.specialist_threshold, 0.0);
        std::env::remove_var("HYBRID_THRESHOLD");
    }
}
