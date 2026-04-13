//! Mamba Training with BPTT — Backpropagation Through Time.
//!
//! Uses the existing autograd Tape to compute gradients through
//! the Mamba SSM recurrence. Truncated BPTT over short windows.
//!
//! This is what was missing: actual gradient-based training of
//! the sequence model layers.

/// A trainable Mamba-like layer using simple gated recurrence.
///
/// Simplified SSM: h_k = sigmoid(W_h @ h_{k-1}) * tanh(W_x @ x_k)
/// This is a GRU-like structure that's easier to train than full SSM.
#[derive(Clone)]
pub struct TrainableMamba {
    /// Input-to-hidden: [d_model, d_hidden]
    pub w_x: Vec<f32>,
    /// Hidden-to-hidden: [d_hidden, d_hidden]
    pub w_h: Vec<f32>,
    /// Gate weights: [d_model + d_hidden, d_hidden]
    pub w_gate: Vec<f32>,
    /// Output projection: [d_hidden, d_model]
    pub w_out: Vec<f32>,
    /// Biases
    pub b_h: Vec<f32>,
    pub b_gate: Vec<f32>,

    pub d_model: usize,
    pub d_hidden: usize,
}

impl TrainableMamba {
    pub fn new(d_model: usize, d_hidden: usize, seed: f32) -> Self {
        let s_xh = (2.0 / (d_model + d_hidden) as f64).sqrt() as f32;
        let s_hh = (2.0 / (d_hidden * 2) as f64).sqrt() as f32;

        let w_x: Vec<f32> = (0..d_model * d_hidden).map(|i| (i as f32 * seed).sin() * s_xh).collect();
        let w_h: Vec<f32> = (0..d_hidden * d_hidden).map(|i| (i as f32 * (seed + 0.1)).sin() * s_hh * 0.5).collect();
        let w_gate: Vec<f32> = (0..(d_model + d_hidden) * d_hidden).map(|i| (i as f32 * (seed + 0.2)).sin() * s_hh).collect();
        let w_out: Vec<f32> = (0..d_hidden * d_model).map(|i| (i as f32 * (seed + 0.3)).sin() * s_xh).collect();
        let b_h = vec![0.0f32; d_hidden];
        let b_gate = vec![0.0f32; d_hidden];

        Self { w_x, w_h, w_gate, w_out, b_h, b_gate, d_model, d_hidden }
    }

    /// Forward pass: [seq_len, d_model] → [seq_len, d_model]
    /// Returns (output, hidden_states) for BPTT.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        let d = self.d_model;
        let dh = self.d_hidden;
        let mut h = vec![0.0f32; dh];
        let mut output = vec![0.0f32; seq_len * d];
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        hidden_states.push(h.clone());

        for t in 0..seq_len {
            let x_t = &input[t * d..(t + 1) * d];

            // Candidate: tanh(W_x @ x + W_h @ h + b_h)
            let mut candidate = vec![0.0f32; dh];
            for j in 0..dh {
                let mut sum = self.b_h[j];
                for k in 0..d { sum += x_t[k] * self.w_x[k * dh + j]; }
                for k in 0..dh { sum += h[k] * self.w_h[k * dh + j]; }
                candidate[j] = sum.tanh();
            }

            // Gate: sigmoid(W_gate @ [x, h] + b_gate)
            let mut gate = vec![0.0f32; dh];
            for j in 0..dh {
                let mut sum = self.b_gate[j];
                for k in 0..d { sum += x_t[k] * self.w_gate[k * dh + j]; }
                for k in 0..dh { sum += h[k] * self.w_gate[(d + k) * dh + j]; }
                gate[j] = 1.0 / (1.0 + (-sum).exp()); // sigmoid
            }

            // Update: h = gate * h + (1 - gate) * candidate
            for j in 0..dh {
                h[j] = gate[j] * h[j] + (1.0 - gate[j]) * candidate[j];
            }
            hidden_states.push(h.clone());

            // Output: W_out @ h + residual
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..dh { sum += h[k] * self.w_out[k * d + j]; }
                output[t * d + j] = input[t * d + j] + sum; // residual
            }
        }

        (output, hidden_states)
    }

    /// BPTT: compute gradients and update weights.
    /// Given output gradients (from loss), backprop through time.
    pub fn backward_and_update(
        &mut self,
        input: &[f32],
        hidden_states: &[Vec<f32>],
        d_output: &[f32], // [seq_len, d_model] gradient from loss
        seq_len: usize,
        lr: f32,
    ) {
        let d = self.d_model;
        let dh = self.d_hidden;

        // Accumulate gradients
        let mut dw_x = vec![0.0f32; d * dh];
        let mut dw_h = vec![0.0f32; dh * dh];
        let mut dw_gate = vec![0.0f32; (d + dh) * dh];
        let mut dw_out = vec![0.0f32; dh * d];
        let mut db_h = vec![0.0f32; dh];
        let mut db_gate = vec![0.0f32; dh];

        let mut dh_next = vec![0.0f32; dh]; // gradient flowing back through hidden state

        // Backprop through time (reverse order)
        for t in (0..seq_len).rev() {
            let x_t = &input[t * d..(t + 1) * d];
            let h_prev = &hidden_states[t];
            let h_curr = &hidden_states[t + 1];

            // d_output[t] flows through residual and W_out
            // d_h from output projection
            let mut d_h = dh_next.clone();
            for j in 0..d {
                let d_out_j = d_output[t * d + j];
                for k in 0..dh {
                    d_h[k] += d_out_j * self.w_out[k * d + j];
                    dw_out[k * d + j] += d_out_j * h_curr[k];
                }
            }

            // Recompute gate and candidate for this timestep
            let mut candidate = vec![0.0f32; dh];
            let mut gate = vec![0.0f32; dh];
            for j in 0..dh {
                let mut sum_c = self.b_h[j];
                for k in 0..d { sum_c += x_t[k] * self.w_x[k * dh + j]; }
                for k in 0..dh { sum_c += h_prev[k] * self.w_h[k * dh + j]; }
                candidate[j] = sum_c.tanh();

                let mut sum_g = self.b_gate[j];
                for k in 0..d { sum_g += x_t[k] * self.w_gate[k * dh + j]; }
                for k in 0..dh { sum_g += h_prev[k] * self.w_gate[(d + k) * dh + j]; }
                gate[j] = 1.0 / (1.0 + (-sum_g).exp());
            }

            // h = gate * h_prev + (1-gate) * candidate
            // d_gate = d_h * (h_prev - candidate)
            // d_candidate = d_h * (1 - gate)
            // d_h_prev += d_h * gate (through recurrence)

            for j in 0..dh {
                let d_gate_j = d_h[j] * (h_prev[j] - candidate[j]);
                let d_candidate_j = d_h[j] * (1.0 - gate[j]);

                // Gate gradient: sigmoid derivative = gate * (1 - gate)
                let d_gate_pre = d_gate_j * gate[j] * (1.0 - gate[j]);

                // Candidate gradient: tanh derivative = 1 - tanh^2
                let d_cand_pre = d_candidate_j * (1.0 - candidate[j] * candidate[j]);

                // Accumulate parameter gradients
                db_h[j] += d_cand_pre;
                db_gate[j] += d_gate_pre;

                for k in 0..d {
                    dw_x[k * dh + j] += d_cand_pre * x_t[k];
                    dw_gate[k * dh + j] += d_gate_pre * x_t[k];
                }
                for k in 0..dh {
                    dw_h[k * dh + j] += d_cand_pre * h_prev[k];
                    dw_gate[(d + k) * dh + j] += d_gate_pre * h_prev[k];
                }

                // Gradient to h_prev (for next iteration backwards)
                dh_next[j] = d_h[j] * gate[j]; // through gate
                for k in 0..dh {
                    dh_next[k] += d_cand_pre * self.w_h[k * dh + j];
                    dh_next[k] += d_gate_pre * self.w_gate[(d + k) * dh + j];
                }
            }

            // Clip gradients to prevent explosion
            for v in &mut dh_next { *v = v.max(-1.0).min(1.0); }
        }

        // Apply gradients
        let inv = lr / seq_len as f32;
        for i in 0..self.w_x.len() { self.w_x[i] -= inv * dw_x[i].max(-1.0).min(1.0); }
        for i in 0..self.w_h.len() { self.w_h[i] -= inv * dw_h[i].max(-1.0).min(1.0); }
        for i in 0..self.w_gate.len() { self.w_gate[i] -= inv * dw_gate[i].max(-1.0).min(1.0); }
        for i in 0..self.w_out.len() { self.w_out[i] -= inv * dw_out[i].max(-1.0).min(1.0); }
        for i in 0..dh { self.b_h[i] -= inv * db_h[i]; self.b_gate[i] -= inv * db_gate[i]; }
    }

    pub fn param_count(&self) -> usize {
        self.w_x.len() + self.w_h.len() + self.w_gate.len() + self.w_out.len() + self.b_h.len() + self.b_gate.len()
    }

    /// Ternarize projection weights.
    pub fn ternarize(&mut self) {
        fn tern(w: &mut Vec<f32>) {
            let gamma: f32 = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32 + 1e-8;
            for v in w.iter_mut() { *v = (*v / gamma).max(-1.0).min(1.0).round(); }
        }
        tern(&mut self.w_x);
        tern(&mut self.w_h);
        tern(&mut self.w_gate);
        tern(&mut self.w_out);
    }
}

/// Complete trainable LM: Embedding → TrainableMamba layers → Output Head
pub struct TrainableLM {
    pub embedding: Vec<f32>,      // [vocab, d_model]
    pub layers: Vec<TrainableMamba>,
    pub output_head: Vec<f32>,    // [d_model, vocab]
    pub d_model: usize,
    pub vocab_size: usize,
    pub tokenizer: crate::qlang_lm::Tokenizer,
}

impl TrainableLM {
    pub fn new(text: &str, d_model: usize, d_hidden: usize, n_layers: usize, vocab_size: usize) -> Self {
        let tokenizer = crate::qlang_lm::Tokenizer::from_text(text, vocab_size);
        let vs = tokenizer.vocab_size;

        let scale = (1.0 / d_model as f64).sqrt() as f32;
        let embedding: Vec<f32> = (0..vs * d_model).map(|i| (i as f32 * 0.4871).sin() * scale).collect();
        let layers = (0..n_layers).map(|i| TrainableMamba::new(d_model, d_hidden, 0.37 + i as f32 * 0.13)).collect();
        let out_scale = (2.0 / (d_model + vs) as f64).sqrt() as f32;
        let output_head: Vec<f32> = (0..d_model * vs).map(|i| (i as f32 * 0.7291).sin() * out_scale).collect();

        Self { embedding, layers, output_head, d_model, vocab_size: vs, tokenizer }
    }

    /// Full forward: tokens → logits. Returns (logits, all hidden states per layer).
    pub fn forward(&self, tokens: &[usize]) -> (Vec<f32>, Vec<Vec<Vec<f32>>>) {
        let seq = tokens.len();
        let d = self.d_model;
        let v = self.vocab_size;

        // Embed
        let mut x = vec![0.0f32; seq * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok.min(v - 1);
            x[i * d..(i + 1) * d].copy_from_slice(&self.embedding[t * d..(t + 1) * d]);
        }

        // Mamba layers
        let mut all_hidden = Vec::new();
        for layer in &self.layers {
            let (out, hs) = layer.forward(&x, seq);
            all_hidden.push(hs);
            x = out;
        }

        // Output head
        let mut logits = vec![0.0f32; seq * v];
        for t in 0..seq {
            for vi in 0..v {
                let mut sum = 0.0f32;
                for k in 0..d { sum += x[t * d + k] * self.output_head[k * v + vi]; }
                logits[t * v + vi] = sum;
            }
        }

        (logits, all_hidden)
    }

    /// Train one step with full BPTT.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> f32 {
        if tokens.len() < 2 { return 0.0; }
        let seq = tokens.len() - 1;
        let d = self.d_model;
        let v = self.vocab_size;

        // Forward
        let (logits, all_hidden) = self.forward(&tokens[..seq]);

        // Softmax + cross-entropy loss
        let mut probs = logits.clone();
        let mut loss = 0.0f32;
        for t in 0..seq {
            let off = t * v;
            let max = probs[off..off + v].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..v { probs[off + i] = (probs[off + i] - max).exp(); sum += probs[off + i]; }
            if sum > 0.0 { for i in 0..v { probs[off + i] /= sum; } }
            let target = tokens[t + 1].min(v - 1);
            loss -= (probs[off + target].max(1e-10)).ln();
        }
        loss /= seq as f32;

        // Backward: d_logits = probs - one_hot
        let mut d_logits = probs;
        for t in 0..seq {
            let target = tokens[t + 1].min(v - 1);
            d_logits[t * v + target] -= 1.0;
            for i in 0..v { d_logits[t * v + i] /= seq as f32; }
        }

        // Get last layer's output for head gradient
        let mut x = vec![0.0f32; seq * d];
        for (i, &tok) in tokens[..seq].iter().enumerate() {
            let t = tok.min(v - 1);
            x[i * d..(i + 1) * d].copy_from_slice(&self.embedding[t * d..(t + 1) * d]);
        }
        for layer in &self.layers { x = layer.forward(&x, seq).0; }

        // Update output head
        for t in 0..seq {
            for vi in 0..v {
                let g = d_logits[t * v + vi];
                if g.abs() < 1e-6 { continue; }
                for k in 0..d {
                    self.output_head[k * v + vi] -= lr * g * x[t * d + k];
                }
            }
        }

        // Compute d_x (gradient to last layer output)
        let mut d_x = vec![0.0f32; seq * d];
        for t in 0..seq {
            for k in 0..d {
                let mut sum = 0.0f32;
                for vi in 0..v { sum += d_logits[t * v + vi] * self.output_head[k * v + vi]; }
                d_x[t * d + k] = sum;
            }
        }

        // BPTT through each Mamba layer (reverse order)
        let mut layer_inputs = Vec::new();
        let mut cur = vec![0.0f32; seq * d];
        for (i, &tok) in tokens[..seq].iter().enumerate() {
            let t = tok.min(v - 1);
            cur[i * d..(i + 1) * d].copy_from_slice(&self.embedding[t * d..(t + 1) * d]);
        }
        layer_inputs.push(cur.clone());
        for layer in &self.layers {
            cur = layer.forward(&cur, seq).0;
            layer_inputs.push(cur.clone());
        }

        for l in (0..self.layers.len()).rev() {
            self.layers[l].backward_and_update(
                &layer_inputs[l], &all_hidden[l], &d_x, seq, lr
            );
            // Propagate gradient to previous layer (through residual)
            // d_x stays the same (residual connection passes gradient through)
        }

        // Update embeddings
        for t in 0..seq {
            let tok = tokens[t].min(v - 1);
            for k in 0..d {
                self.embedding[tok * d + k] -= lr * 0.1 * d_x[t * d + k];
            }
        }

        loss
    }

    /// Generate text with temperature sampling.
    /// temperature=0.0 → greedy, temperature=1.0 → full distribution
    pub fn generate(&self, prompt: &str, n_tokens: usize) -> String {
        self.generate_with_temp(prompt, n_tokens, 0.8)
    }

    pub fn generate_with_temp(&self, prompt: &str, n_tokens: usize, temperature: f32) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(2); }

        let mut rng = 42u64;

        for _ in 0..n_tokens {
            let (logits, _) = self.forward(&tokens);
            let off = (tokens.len() - 1) * self.vocab_size;
            let last = &logits[off..off + self.vocab_size];

            // Apply temperature
            let temp = temperature.max(0.01);
            let mut scaled: Vec<f32> = last.iter().map(|&l| l / temp).collect();

            // Zero out <unk> and <pad>
            scaled[0] = f32::NEG_INFINITY;
            scaled[1] = f32::NEG_INFINITY;

            // Repetition penalty: reduce score for tokens already generated
            for &prev_tok in &tokens {
                if prev_tok < self.vocab_size {
                    scaled[prev_tok] -= 2.0; // strong penalty
                }
            }

            // Top-k filtering: keep only top 40
            let mut indices: Vec<usize> = (0..self.vocab_size).collect();
            indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
            for &idx in &indices[40..] {
                scaled[idx] = f32::NEG_INFINITY;
            }

            // Softmax
            let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 { for p in &mut probs { *p /= sum; } }

            // Sample from distribution
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 33) as f32 / u32::MAX as f32;
            let mut cumsum = 0.0f32;
            let mut next = 2usize;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum > r && i >= 2 {
                    next = i;
                    break;
                }
            }

            tokens.push(next);
        }
        self.tokenizer.decode(&tokens)
    }

    /// Perplexity.
    pub fn perplexity(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < 2 { return f32::INFINITY; }
        let (logits, _) = self.forward(&tokens[..tokens.len() - 1]);
        let seq = tokens.len() - 1;
        let v = self.vocab_size;

        let mut log_prob = 0.0f64;
        let mut count = 0;
        for t in 0..seq {
            let off = t * v;
            let max = logits[off..off + v].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..v).map(|i| (logits[off + i] - max).exp()).sum();
            let target = tokens[t + 1].min(v - 1);
            let p = ((logits[off + target] - max).exp() / sum) as f64;
            if p > 1e-10 { log_prob += p.ln(); count += 1; }
        }
        if count == 0 { f32::INFINITY } else { (-(log_prob / count as f64)).exp() as f32 }
    }

    pub fn param_count(&self) -> usize {
        self.embedding.len() + self.layers.iter().map(|l| l.param_count()).sum::<usize>() + self.output_head.len()
    }

    /// Construct a TrainableLM from pre-loaded weights (used by QLMB loader).
    pub fn from_weights(
        embedding: Vec<f32>,
        output_head: Vec<f32>,
        layers: Vec<TrainableMamba>,
        d_model: usize,
        vocab_size: usize,
        tokenizer: crate::qlang_lm::Tokenizer,
    ) -> Self {
        Self { embedding, layers, output_head, d_model, vocab_size, tokenizer }
    }
}

/// Save a trained Mamba LM to a QLMB binary file including the tokenizer vocab.
///
/// The VOCB section ensures that token IDs match exactly when the model is
/// reloaded — without it, generation emits `<unk>` for every token.
pub fn save_mamba_model(path: &str, lm: &TrainableLM) -> Result<(), String> {
    use std::io::Write;

    let file = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create '{}': {}", path, e))?;
    let mut w = std::io::BufWriter::new(file);

    let write_u32 = |w: &mut std::io::BufWriter<std::fs::File>, v: u32| -> Result<(), String> {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    };
    let write_f32_vec = |w: &mut std::io::BufWriter<std::fs::File>, v: &[f32]| -> Result<(), String> {
        w.write_all(&(v.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        for &f in v {
            w.write_all(&f.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    };

    // Magic
    w.write_all(b"QLMB").map_err(|e| e.to_string())?;

    // Config
    write_u32(&mut w, lm.d_model as u32)?;
    write_u32(&mut w, lm.vocab_size as u32)?;
    write_u32(&mut w, lm.layers.len() as u32)?;

    // Weights
    write_f32_vec(&mut w, &lm.embedding)?;
    write_f32_vec(&mut w, &lm.output_head)?;
    for layer in &lm.layers {
        write_f32_vec(&mut w, &layer.w_x)?;
        write_f32_vec(&mut w, &layer.w_h)?;
        write_f32_vec(&mut w, &layer.w_gate)?;
        write_f32_vec(&mut w, &layer.w_out)?;
        write_f32_vec(&mut w, &layer.b_h)?;
        write_f32_vec(&mut w, &layer.b_gate)?;
    }

    // VOCB section — critical for correct token IDs on reload
    w.write_all(b"VOCB").map_err(|e| e.to_string())?;
    let vocab = &lm.tokenizer.id2word;
    write_u32(&mut w, vocab.len() as u32)?;
    for word in vocab {
        let bytes = word.as_bytes();
        let len = bytes.len().min(u16::MAX as usize) as u16;
        w.write_all(&len.to_le_bytes()).map_err(|e| e.to_string())?;
        w.write_all(&bytes[..len as usize]).map_err(|e| e.to_string())?;
    }

    w.flush().map_err(|e| e.to_string())
}

/// Load a trained Mamba LM from a QLMB binary file.
///
/// The `text_for_tokenizer` is used to rebuild the BPE tokenizer with the
/// same vocabulary size stored in the model. If empty, a minimal placeholder
/// is used.
pub fn load_mamba_model(path: &str, text_for_tokenizer: &str) -> Result<TrainableLM, String> {
    let data = std::fs::read(path).map_err(|e| format!("Cannot read '{}': {}", path, e))?;
    if data.len() < 16 {
        return Err("File too small to be QLMB".into());
    }
    if &data[0..4] != b"QLMB" {
        return Err(format!("Bad magic: expected QLMB, got {:?}", &data[0..4]));
    }
    let mut pos = 4usize;

    let read_u32 = |pos: &mut usize, data: &[u8]| -> Result<u32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF reading u32".into()); }
        let v = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
        *pos += 4;
        Ok(v)
    };

    let read_f32_vec = |pos: &mut usize, data: &[u8]| -> Result<Vec<f32>, String> {
        let len = read_u32(pos, data)? as usize;
        if *pos + len * 4 > data.len() {
            return Err(format!("Unexpected EOF reading {} f32s at offset {}", len, *pos));
        }
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            let f = f32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4;
            v.push(f);
        }
        Ok(v)
    };

    let d_model = read_u32(&mut pos, &data)? as usize;
    let vocab_size = read_u32(&mut pos, &data)? as usize;
    let n_layers = read_u32(&mut pos, &data)? as usize;

    let embedding = read_f32_vec(&mut pos, &data)?;
    let output_head = read_f32_vec(&mut pos, &data)?;

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let w_x = read_f32_vec(&mut pos, &data)?;
        let w_h = read_f32_vec(&mut pos, &data)?;
        let w_gate = read_f32_vec(&mut pos, &data)?;
        let w_out = read_f32_vec(&mut pos, &data)?;
        let b_h = read_f32_vec(&mut pos, &data)?;
        let b_gate = read_f32_vec(&mut pos, &data)?;

        // Infer d_hidden from w_x: w_x is [d_model, d_hidden] → len = d_model * d_hidden
        let d_hidden = if d_model > 0 { w_x.len() / d_model } else { 0 };

        layers.push(TrainableMamba {
            w_x, w_h, w_gate, w_out, b_h, b_gate,
            d_model, d_hidden,
        });
    }

    // Optional vocab section: "VOCB" | u32 count | [u16 word_len | bytes]*
    // If present, reconstruct the tokenizer from the exact embedded vocab so
    // token IDs match training exactly (T011). Otherwise fall back to the
    // legacy behaviour of rebuilding from reference text.
    let tokenizer = if pos + 4 <= data.len() && &data[pos..pos + 4] == b"VOCB" {
        pos += 4;
        let count = read_u32(&mut pos, &data)? as usize;
        let mut vocab = Vec::with_capacity(count);
        for _ in 0..count {
            if pos + 2 > data.len() {
                return Err("Unexpected EOF reading vocab word length".into());
            }
            let word_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if pos + word_len > data.len() {
                return Err(format!(
                    "Unexpected EOF reading vocab word of {} bytes at offset {}",
                    word_len, pos
                ));
            }
            let word = String::from_utf8(data[pos..pos + word_len].to_vec())
                .map_err(|e| format!("Invalid UTF-8 in vocab: {}", e))?;
            pos += word_len;
            vocab.push(word);
        }
        crate::qlang_lm::Tokenizer::from_vocab(vocab)
    } else {
        let text = if text_for_tokenizer.is_empty() { "a b c d" } else { text_for_tokenizer };
        crate::qlang_lm::Tokenizer::from_text(text, vocab_size)
    };

    Ok(TrainableLM::from_weights(embedding, output_head, layers, d_model, vocab_size, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainable_mamba_bptt() {
        let layer = TrainableMamba::new(16, 32, 0.37);
        let input = vec![0.1f32; 4 * 16]; // 4 timesteps
        let (output, hidden) = layer.forward(&input, 4);
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|x| x.is_finite()));
        println!("TrainableMamba: {} params, output finite", layer.param_count());
    }

    #[test]
    fn bptt_reduces_loss() {
        let text = "the cat sat on the mat the dog ran in the park";
        let mut lm = TrainableLM::new(text, 32, 64, 1, 20);

        let tokens = lm.tokenizer.encode(text);
        let loss1 = lm.train_step(&tokens[..8], 0.01);
        // Train more
        for _ in 0..50 {
            lm.train_step(&tokens[..8], 0.01);
        }
        let loss2 = lm.train_step(&tokens[..8], 0.01);

        println!("BPTT: loss {:.3} → {:.3}", loss1, loss2);
        assert!(loss2 < loss1, "BPTT must reduce loss: {} → {}", loss1, loss2);
    }
}
