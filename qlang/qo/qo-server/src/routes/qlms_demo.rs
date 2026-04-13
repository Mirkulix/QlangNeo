//! QLMS AI-to-AI round-trip demo.
//!
//! Binary protocol for exchanging signed TernaryBrain models between QO servers.
//!
//! Wire format (QLMS v1 demo payload):
//!   [4]  magic "QLMS"
//!   [2]  version (LE u16) = 1
//!   [2]  kind   (LE u16)  = 0x0001 (MODEL_TERNARY_BRAIN)
//!   [32] hmac-sha256(secret, payload)
//!   [4]  payload_len (LE u32)
//!   [..] payload
//!
//! Payload:
//!   [2]  specialist_id_len (LE u16)
//!   [..] specialist_id (utf-8)
//!   [4]  image_dim (LE u32)
//!   [4]  n_classes (LE u32)
//!   [4]  total_weights (LE u32)
//!   [2]  n_class_names (LE u16)
//!   for each class name: [2] len, [..] utf-8
//!   [8]  timestamp_ms (LE u64)
//!   [total_weights] ternary weights as i8 (-1, 0, +1)
//!
//! Security: every message is HMAC-SHA256 signed with a shared secret
//! (env `QLMS_DEMO_SECRET`, default b"qlang-demo-2026").

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    Json,
};
use once_cell::sync::Lazy;
use qlang_core::crypto::{ct_eq, hex, hmac_sha256, sha256};
use qlang_runtime::ternary_brain::TernaryBrain;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::AppState;

const MAGIC: &[u8; 4] = b"QLMS";
const VERSION: u16 = 1;
const KIND_MODEL: u16 = 0x0001;

// ============================================================
// Shared secret (32-byte key derived via SHA-256 from env)
// ============================================================

fn shared_key() -> [u8; 32] {
    let secret = std::env::var("QLMS_DEMO_SECRET")
        .unwrap_or_else(|_| "qlang-demo-2026".to_string());
    sha256(secret.as_bytes())
}

// ============================================================
// In-memory ring buffer log (last 50 exchanges)
// ============================================================

#[derive(Clone, Serialize)]
pub struct QlmsLogEntry {
    pub ts_ms: u64,
    pub direction: String,  // "send" | "receive"
    pub target_or_source: String,
    pub specialist_id: String,
    pub bytes: usize,
    pub signature_hex: String,
    pub verified: Option<bool>,
    pub prediction: Option<u32>,
    pub confidence: Option<f32>,
    pub round_trip_ms: Option<u128>,
}

static LOG: Lazy<Arc<Mutex<VecDeque<QlmsLogEntry>>>> =
    Lazy::new(|| Arc::new(Mutex::new(VecDeque::with_capacity(64))));

fn log_push(entry: QlmsLogEntry) {
    let mut log = LOG.lock().unwrap();
    if log.len() >= 50 {
        log.pop_front();
    }
    log.push_back(entry);
}

// ============================================================
// Brain bank — stores TernaryBrains by id (so server A can cache
// the trained model and reference it by specialist_id).
// ============================================================

static BRAIN_BANK: Lazy<Arc<Mutex<std::collections::HashMap<String, Arc<TernaryBrain>>>>> =
    Lazy::new(|| Arc::new(Mutex::new(std::collections::HashMap::new())));

fn bank_store(id: &str, brain: TernaryBrain) {
    BRAIN_BANK
        .lock()
        .unwrap()
        .insert(id.to_string(), Arc::new(brain));
}

#[allow(dead_code)]
fn bank_get(id: &str) -> Option<Arc<TernaryBrain>> {
    BRAIN_BANK.lock().unwrap().get(id).cloned()
}

// ============================================================
// Payload encoding / decoding (pure binary, no JSON)
// ============================================================

struct ModelPayload {
    specialist_id: String,
    image_dim: u32,
    n_classes: u32,
    class_names: Vec<String>,
    timestamp_ms: u64,
    weights: Vec<i8>,
}

fn encode_payload(p: &ModelPayload) -> Vec<u8> {
    let total_weights = p.weights.len() as u32;
    let mut buf = Vec::with_capacity(64 + p.specialist_id.len() + p.weights.len());
    let id_bytes = p.specialist_id.as_bytes();
    buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf.extend_from_slice(&p.image_dim.to_le_bytes());
    buf.extend_from_slice(&p.n_classes.to_le_bytes());
    buf.extend_from_slice(&total_weights.to_le_bytes());
    buf.extend_from_slice(&(p.class_names.len() as u16).to_le_bytes());
    for name in &p.class_names {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        buf.extend_from_slice(nb);
    }
    buf.extend_from_slice(&p.timestamp_ms.to_le_bytes());
    // Transmit i8 weights as u8 bytes (same bit pattern, safe reinterpretation)
    let wbytes: Vec<u8> = p.weights.iter().map(|&b| b as u8).collect();
    buf.extend_from_slice(&wbytes);
    buf
}

fn decode_payload(data: &[u8]) -> Result<ModelPayload, String> {
    let mut o = 0usize;
    fn take<'a>(data: &'a [u8], o: &mut usize, n: usize) -> Result<&'a [u8], String> {
        if *o + n > data.len() {
            return Err(format!("truncated at offset {}, need {}", *o, n));
        }
        let s = &data[*o..*o + n];
        *o += n;
        Ok(s)
    }
    let id_len = u16::from_le_bytes(take(data, &mut o, 2)?.try_into().unwrap()) as usize;
    let id = String::from_utf8(take(data, &mut o, id_len)?.to_vec())
        .map_err(|e| format!("bad utf8 in id: {e}"))?;
    let image_dim = u32::from_le_bytes(take(data, &mut o, 4)?.try_into().unwrap());
    let n_classes = u32::from_le_bytes(take(data, &mut o, 4)?.try_into().unwrap());
    let total_weights = u32::from_le_bytes(take(data, &mut o, 4)?.try_into().unwrap());
    let n_names = u16::from_le_bytes(take(data, &mut o, 2)?.try_into().unwrap()) as usize;
    let mut class_names = Vec::with_capacity(n_names);
    for _ in 0..n_names {
        let nlen = u16::from_le_bytes(take(data, &mut o, 2)?.try_into().unwrap()) as usize;
        let nm = String::from_utf8(take(data, &mut o, nlen)?.to_vec())
            .map_err(|e| format!("bad utf8 in class: {e}"))?;
        class_names.push(nm);
    }
    let timestamp_ms = u64::from_le_bytes(take(data, &mut o, 8)?.try_into().unwrap());
    let w_bytes = take(data, &mut o, total_weights as usize)?;
    let weights: Vec<i8> = w_bytes.iter().map(|&b| b as i8).collect();
    Ok(ModelPayload {
        specialist_id: id,
        image_dim,
        n_classes,
        class_names,
        timestamp_ms,
        weights,
    })
}

fn encode_qlms(kind: u16, payload: &[u8]) -> (Vec<u8>, [u8; 32]) {
    let sig = hmac_sha256(&shared_key(), payload);
    let mut buf = Vec::with_capacity(4 + 2 + 2 + 32 + 4 + payload.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&kind.to_le_bytes());
    buf.extend_from_slice(&sig);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    (buf, sig)
}

fn decode_qlms(data: &[u8]) -> Result<(u16, [u8; 32], Vec<u8>), String> {
    if data.len() < 4 + 2 + 2 + 32 + 4 {
        return Err(format!("QLMS frame too short: {} bytes", data.len()));
    }
    if &data[0..4] != MAGIC {
        return Err("bad magic (expected QLMS)".into());
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != VERSION {
        return Err(format!("unsupported QLMS version {}", version));
    }
    let kind = u16::from_le_bytes([data[6], data[7]]);
    let mut sig = [0u8; 32];
    sig.copy_from_slice(&data[8..40]);
    let payload_len = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
    if data.len() < 44 + payload_len {
        return Err(format!(
            "payload truncated: have {}, need {}",
            data.len() - 44,
            payload_len
        ));
    }
    let payload = data[44..44 + payload_len].to_vec();

    // Verify HMAC (constant-time — prevents timing attacks on the tag)
    let expected = hmac_sha256(&shared_key(), &payload);
    if !ct_eq(&expected, &sig) {
        return Err("HMAC verification failed".into());
    }
    Ok((kind, sig, payload))
}

// ============================================================
// Training helper — build a tiny TernaryBrain on synthetic data
// so the demo works without external files.
// ============================================================

fn train_demo_brain() -> (TernaryBrain, u32, u32, Vec<String>, Vec<f32>, u8) {
    // 8x8 mini images, 3 classes (circle, line, corner), 64 dims
    let image_dim: usize = 64;
    let n_classes: usize = 3;
    let n_per_class: usize = 30;
    let n_samples = n_classes * n_per_class;
    let mut images = vec![0.0f32; n_samples * image_dim];
    let mut labels = vec![0u8; n_samples];

    // Deterministic PRNG (xorshift) for reproducibility
    let mut x: u64 = 0xC0DE_FACE_CAFE_1234;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    for (sample_idx, label_idx) in (0..n_samples).map(|i| (i, i / n_per_class)).collect::<Vec<_>>() {
        labels[sample_idx] = label_idx as u8;
        let off = sample_idx * image_dim;
        // Base pattern per class
        for row in 0..8 {
            for col in 0..8 {
                let pixel = off + row * 8 + col;
                let base = match label_idx {
                    0 => {
                        // Circle pattern
                        let dx = col as f32 - 3.5;
                        let dy = row as f32 - 3.5;
                        let d = (dx * dx + dy * dy).sqrt();
                        if (d - 2.5).abs() < 1.0 { 1.0 } else { -0.5 }
                    }
                    1 => {
                        // Horizontal line
                        if row == 4 { 1.0 } else { -0.5 }
                    }
                    _ => {
                        // Corner (top-left)
                        if row < 3 && col < 3 { 1.0 } else { -0.5 }
                    }
                };
                images[pixel] = base + 0.15 * rnd();
            }
        }
    }

    // Train a small brain
    let mut brain = TernaryBrain::init(
        &images, &labels, image_dim, n_samples, n_classes, 8,
    );
    brain.refine(&images, &labels, n_samples, 5);

    // Build a fixed test sample for class 1 (horizontal line) — this is our "digit 7"
    let test_label: u8 = 1;
    let mut test = vec![-0.5f32; image_dim];
    for col in 0..8 {
        test[4 * 8 + col] = 1.0;
    }

    let class_names = vec![
        "circle".to_string(),
        "line".to_string(),
        "corner".to_string(),
    ];
    (brain, image_dim as u32, n_classes as u32, class_names, test, test_label)
}

// Encode a fixed "test digit" alongside the model — appended as a supplementary
// f32 slice after weights, only for the demo. We instead inline via constants on
// the receive side (keeping payload focused on weights).

// ============================================================
// Security: SSRF guard
// ============================================================

/// Validate that a `host[:port]` string is safe to connect to.
///
/// Blocks localhost, link-local, and RFC-1918 private ranges to prevent
/// Server-Side Request Forgery (SSRF) attacks.
fn validate_target_host(host: &str) -> Result<(), String> {
    // Strip optional port suffix
    let hostname = host.split(':').next().unwrap_or("").to_lowercase();

    if hostname.is_empty() {
        return Err("target_host must not be empty".into());
    }

    // Reject URL schemes / path characters
    if host.contains('/') || host.contains('@') || host.contains('#') {
        return Err("target_host must be host[:port] only".into());
    }

    // Reject loopback / localhost names
    let blocked_names = ["localhost", "::1", "0.0.0.0"];
    if blocked_names.contains(&hostname.as_str()) {
        return Err(format!("target_host '{}' is not allowed", hostname));
    }

    // Reject IPv4 private / loopback ranges
    if let Ok(addr) = hostname.parse::<std::net::Ipv4Addr>() {
        if addr.is_loopback() || addr.is_private() || addr.is_link_local() || addr.is_unspecified() {
            return Err(format!("target_host '{}' resolves to a blocked range", hostname));
        }
    }

    // Reject IPv6 loopback
    if let Ok(addr) = hostname.parse::<std::net::Ipv6Addr>() {
        if addr.is_loopback() || addr.is_unspecified() {
            return Err(format!("target_host '{}' is not allowed", hostname));
        }
    }

    // Reject 127.x.x.x range explicitly (loopback is only 127.0.0.1 via is_loopback())
    if hostname.starts_with("127.") || hostname.starts_with("169.254.") {
        return Err(format!("target_host '{}' is not allowed", hostname));
    }

    Ok(())
}

// ============================================================
// HTTP handlers
// ============================================================

#[derive(Deserialize)]
pub struct SendModelInput {
    pub target_host: String,
    #[serde(default)]
    pub specialist_id: Option<String>,
}

#[derive(Serialize)]
pub struct SendModelOutput {
    pub sent: bool,
    pub bytes: usize,
    pub signature: String,
    pub specialist_id: String,
    pub target_ack: serde_json::Value,
}

/// POST /api/qlms/send-model
pub async fn send_model(
    State(_state): State<std::sync::Arc<AppState>>,
    Json(input): Json<SendModelInput>,
) -> Result<Json<SendModelOutput>, (StatusCode, String)> {
    let t0 = Instant::now();
    // Train a small demo brain on this server (server A's "specialist")
    let (brain, image_dim, n_classes, class_names, _test, _label) = train_demo_brain();
    let _total_weights = brain.total_weights();
    let weights_i8 = brain.dump_weights_i8();

    let specialist_id = input
        .specialist_id
        .unwrap_or_else(|| format!("demo-classifier-{}", now_ms()));

    // Cache in bank (so repeated sends could reuse it — useful for tests)
    bank_store(&specialist_id, brain);

    let payload = ModelPayload {
        specialist_id: specialist_id.clone(),
        image_dim,
        n_classes,
        class_names,
        timestamp_ms: now_ms(),
        weights: weights_i8,
    };
    let payload_bytes = encode_payload(&payload);
    let (frame, signature) = encode_qlms(KIND_MODEL, &payload_bytes);
    let sig_hex = hex(&signature);
    let bytes = frame.len();

    // SSRF guard: validate target before making any outbound request
    validate_target_host(&input.target_host)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    // POST to target
    let target_url = format!("http://{}/api/qlms/receive", input.target_host);
    let client = reqwest::Client::new();
    let resp = client
        .post(&target_url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("send failed: {e}")))?;

    let status = resp.status();
    let ack: serde_json::Value = resp.json().await.unwrap_or(serde_json::json!({}));
    if !status.is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("target returned {}: {}", status, ack),
        ));
    }

    let rt_ms = t0.elapsed().as_millis();
    log_push(QlmsLogEntry {
        ts_ms: now_ms(),
        direction: "send".into(),
        target_or_source: input.target_host.clone(),
        specialist_id: specialist_id.clone(),
        bytes,
        signature_hex: sig_hex.clone(),
        verified: ack.get("verified").and_then(|v| v.as_bool()),
        prediction: ack.get("prediction").and_then(|v| v.as_u64()).map(|v| v as u32),
        confidence: ack.get("confidence").and_then(|v| v.as_f64()).map(|v| v as f32),
        round_trip_ms: Some(rt_ms),
    });

    tracing::info!(
        "QLMS: sent {} bytes to {}, sig={}…, rtt={}ms",
        bytes,
        input.target_host,
        &sig_hex[..12],
        rt_ms
    );

    Ok(Json(SendModelOutput {
        sent: true,
        bytes,
        signature: sig_hex,
        specialist_id,
        target_ack: ack,
    }))
}

#[derive(Serialize)]
pub struct ReceiveOutput {
    pub verified: bool,
    pub prediction: u32,
    pub confidence: f32,
    pub class_name: String,
    pub round_trip_ms: u128,
    pub bytes: usize,
    pub specialist_id: String,
}

/// POST /api/qlms/receive  (raw binary body)
pub async fn receive_model(
    State(_state): State<std::sync::Arc<AppState>>,
    body: Bytes,
) -> Result<Json<ReceiveOutput>, (StatusCode, String)> {
    let t0 = Instant::now();
    let bytes = body.len();

    let (kind, sig, payload_bytes) = decode_qlms(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("decode/verify failed: {e}")))?;

    if kind != KIND_MODEL {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("unexpected kind 0x{:04x}", kind),
        ));
    }

    let payload = decode_payload(&payload_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("payload decode: {e}")))?;

    // Rebuild the brain. We need a template with matching topology to use
    // `from_template_and_weights`. Build it synthetically with matching dims.
    // (Server B trains its own template the same way — deterministic via seed.)
    let (template, _image_dim, _n_classes, _class_names, test_sample, expected_label) =
        train_demo_brain();

    // Sanity check dims
    if template.image_dim as u32 != payload.image_dim
        || template.n_classes as u32 != payload.n_classes
        || template.total_weights() != payload.weights.len()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "dimension mismatch: got img_dim={} n_classes={} weights={}, expected img_dim={} n_classes={} weights={}",
                payload.image_dim, payload.n_classes, payload.weights.len(),
                template.image_dim, template.n_classes, template.total_weights(),
            ),
        ));
    }

    let brain = TernaryBrain::from_template_and_weights(&template, &payload.weights)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("load weights: {e}")))?;
    // Replace template weights with payload (from_template_and_weights already did)
    // Also validate the brain is fully ternary
    if !brain.verify_ternary() {
        return Err((StatusCode::BAD_REQUEST, "non-ternary weights".into()));
    }

    // Classify fixed test sample
    let preds = brain.predict(&test_sample, 1);
    let prediction = *preds.first().unwrap_or(&0) as u32;

    // Confidence = max normalized score (heuristic)
    let scores = brain.layer.predict_one(&test_sample, brain.n_classes);
    let max_s = *scores.iter().max().unwrap_or(&1) as f32;
    let sum_s = scores.iter().map(|&v| v.max(0) as f32).sum::<f32>().max(1.0);
    let confidence = (max_s.max(0.0) / sum_s).min(1.0);

    let class_name = payload
        .class_names
        .get(prediction as usize)
        .cloned()
        .unwrap_or_else(|| format!("class_{}", prediction));

    // Store received brain
    bank_store(&payload.specialist_id, brain);

    // Force touch expected_label to silence unused if demo simplified
    let _ = expected_label;

    let rt_ms = t0.elapsed().as_millis();
    let sig_hex = hex(&sig);
    log_push(QlmsLogEntry {
        ts_ms: now_ms(),
        direction: "receive".into(),
        target_or_source: "inbound".into(),
        specialist_id: payload.specialist_id.clone(),
        bytes,
        signature_hex: sig_hex.clone(),
        verified: Some(true),
        prediction: Some(prediction),
        confidence: Some(confidence),
        round_trip_ms: Some(rt_ms),
    });

    tracing::info!(
        "QLMS: received {} bytes (verified), specialist={}, pred={} conf={:.2}",
        bytes, payload.specialist_id, prediction, confidence
    );

    Ok(Json(ReceiveOutput {
        verified: true,
        prediction,
        confidence,
        class_name,
        round_trip_ms: rt_ms,
        bytes,
        specialist_id: payload.specialist_id,
    }))
}

/// GET /api/qlms/demo-log
pub async fn demo_log() -> Json<serde_json::Value> {
    let log = LOG.lock().unwrap();
    let entries: Vec<&QlmsLogEntry> = log.iter().collect();
    Json(serde_json::json!({
        "count": entries.len(),
        "entries": entries,
    }))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================
// Tests (decoupled from HTTP — exercise the codec)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_payload() {
        let p = ModelPayload {
            specialist_id: "xyz".into(),
            image_dim: 64,
            n_classes: 3,
            class_names: vec!["a".into(), "b".into(), "c".into()],
            timestamp_ms: 1_234_567_890,
            weights: vec![-1, 0, 1, 1, -1, 0, 0, 1],
        };
        let bytes = encode_payload(&p);
        let d = decode_payload(&bytes).unwrap();
        assert_eq!(d.specialist_id, "xyz");
        assert_eq!(d.image_dim, 64);
        assert_eq!(d.n_classes, 3);
        assert_eq!(d.class_names, vec!["a", "b", "c"]);
        assert_eq!(d.weights, p.weights);
    }

    #[test]
    fn qlms_frame_verifies() {
        let payload = b"hello ternary";
        let (frame, _sig) = encode_qlms(KIND_MODEL, payload);
        let (kind, _sig2, got) = decode_qlms(&frame).unwrap();
        assert_eq!(kind, KIND_MODEL);
        assert_eq!(got, payload);
    }

    #[test]
    fn qlms_rejects_tampered_payload() {
        let payload = b"hello ternary";
        let (mut frame, _sig) = encode_qlms(KIND_MODEL, payload);
        // Tamper with a payload byte
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        let res = decode_qlms(&frame);
        assert!(res.is_err(), "tampered frame must fail HMAC verify");
    }

    #[test]
    fn demo_brain_trains() {
        let (brain, dim, n_classes, names, _test, _label) = train_demo_brain();
        assert_eq!(dim, 64);
        assert_eq!(n_classes, 3);
        assert_eq!(names.len(), 3);
        assert!(brain.verify_ternary());
        assert!(brain.total_weights() > 0);
    }
}
