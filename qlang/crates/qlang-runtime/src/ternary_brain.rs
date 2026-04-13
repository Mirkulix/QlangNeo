//! Ternary Brain — Statistical Init + Competitive Hebbian Refinement.
//!
//! A genuinely new approach combining two proven ideas:
//!
//! Phase 1: STATISTICAL INITIALIZATION
//!   Each neuron's ternary weights are set from class statistics.
//!   This gives 70%+ immediately — no iterative training needed.
//!   (Proven in ternary_ensemble.rs)
//!
//! Phase 2: COMPETITIVE HEBBIAN REFINEMENT (Winner-Take-All)
//!   Neurons COMPETE within each layer. When a sample arrives:
//!   - All neurons compute their activation
//!   - The WINNER (highest activation) gets reinforced
//!   - The LOSERS get suppressed
//!   - Weight updates are ternary flips based on correlation + reward
//!
//! This mimics biological lateral inhibition:
//!   - Neurons specialize (each "owns" a pattern)
//!   - No two neurons learn the same thing
//!   - The competition creates discrimination without gradients
//!
//! Properties:
//!   - 100% ternary at all times (i8 weights)
//!   - No f32 shadow weights
//!   - No gradients, no loss function
//!   - Bootstrap problem solved by statistical initialization
//!   - Refinement is local, online, and biologically plausible

use rayon::prelude::*;

// ============================================================
// Types
// ============================================================

/// A single competitive ternary neuron.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TernaryNeuron {
    /// Ternary weights {-1, 0, +1} as i8
    weights: Vec<i8>,
    /// Bias (integer, scaled by 1000)
    bias: i32,
    /// Which class this neuron currently represents (-1 = unassigned)
    assigned_class: i8,
    /// Win count (how often this neuron was the winner)
    wins: u32,
    /// How many times this neuron's prediction was correct when winning
    correct_wins: u32,
}

impl TernaryNeuron {
    /// Compute activation score: pure add/sub, no multiply.
    #[inline]
    fn score(&self, x: &[f32]) -> i32 {
        let mut sum = self.bias;
        for k in 0..self.weights.len() {
            match self.weights[k] {
                1 => sum += (x[k] * 1000.0) as i32,
                -1 => sum -= (x[k] * 1000.0) as i32,
                _ => {}
            }
        }
        sum
    }

    /// Hebbian update: conservative — only flip weights with very strong signal.
    fn hebbian_update(&mut self, x: &[f32], reward: bool, flip_rate: f32) {
        // Only update a small fraction of weights per step
        let n_candidates = (self.weights.len() as f32 * flip_rate) as usize;
        let mut updated = 0;

        for k in 0..self.weights.len() {
            if updated >= n_candidates { break; }
            let xk = x[k];
            if xk.abs() < 0.3 { continue; } // only strong signals

            let input_sign = if xk > 0.0 { 1i8 } else { -1i8 };

            if reward && self.weights[k] == 0 {
                // Strengthen only zero → nonzero (never flip existing)
                self.weights[k] = input_sign;
                updated += 1;
            } else if !reward && self.weights[k] == input_sign {
                // Weaken: nonzero → zero (never flip sign)
                self.weights[k] = 0;
                updated += 1;
            }
        }
    }
}

// ============================================================
// Competitive Layer
// ============================================================

/// A layer of competing ternary neurons (Winner-Take-All).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompetitiveLayer {
    neurons: Vec<TernaryNeuron>,
    in_dim: usize,
    n_neurons: usize,
}

impl CompetitiveLayer {
    /// Initialize from class statistics using SUBSET DIVERSITY.
    ///
    /// Instead of varying thresholds on the same mean, each neuron is derived
    /// from a DIFFERENT SUBSET of training samples. This creates genuine diversity:
    /// neuron 1 captures one variation of digit "3", neuron 2 captures another.
    ///
    /// This is ternary k-means: cluster within each class, then ternarize each cluster center.
    pub fn from_statistics(
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        n_classes: usize,
        neurons_per_class: usize,
    ) -> Self {
        let _n_neurons = n_classes * neurons_per_class;

        // Collect indices per class
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for i in 0..n_samples {
            let c = labels[i] as usize;
            if c < n_classes { class_indices[c].push(i); }
        }

        // Global mean for bias computation
        let mut global_mean = vec![0.0f64; image_dim];
        for i in 0..n_samples {
            for k in 0..image_dim {
                global_mean[k] += images[i * image_dim + k] as f64;
            }
        }
        for k in 0..image_dim { global_mean[k] /= n_samples as f64; }

        // Parallel neuron creation per class
        let neurons: Vec<TernaryNeuron> = (0..n_classes)
            .into_par_iter()
            .flat_map(|c| {
            let indices = &class_indices[c];
            let n_per_subset = indices.len() / neurons_per_class.max(1);
            let mut class_neurons = Vec::with_capacity(neurons_per_class);

            for variant in 0..neurons_per_class {
                let start = variant * n_per_subset;
                let end = if variant == neurons_per_class - 1 {
                    indices.len()
                } else {
                    (variant + 1) * n_per_subset
                };
                let subset = &indices[start.min(indices.len())..end.min(indices.len())];
                if subset.is_empty() { continue; }

                let mut subset_mean = vec![0.0f64; image_dim];
                for &idx in subset {
                    for k in 0..image_dim {
                        subset_mean[k] += images[idx * image_dim + k] as f64;
                    }
                }
                for k in 0..image_dim { subset_mean[k] /= subset.len() as f64; }

                let diff: Vec<f64> = (0..image_dim)
                    .map(|k| subset_mean[k] - global_mean[k])
                    .collect();
                let mean_abs: f64 = diff.iter().map(|d| d.abs()).sum::<f64>() / image_dim as f64;
                let threshold = mean_abs * 0.5;

                let weights: Vec<i8> = diff.iter().map(|&d| {
                    if d > threshold { 1i8 } else if d < -threshold { -1i8 } else { 0i8 }
                }).collect();

                let class_score: i32 = weights.iter().zip(subset_mean.iter())
                    .map(|(&w, &m)| w as i32 * (m * 1000.0) as i32)
                    .sum();
                let global_score: i32 = weights.iter().zip(global_mean.iter())
                    .map(|(&w, &m)| w as i32 * (m * 1000.0) as i32)
                    .sum();
                let bias = -(class_score + global_score) / 2;

                class_neurons.push(TernaryNeuron {
                    weights, bias,
                    assigned_class: c as i8,
                    wins: 0, correct_wins: 0,
                });
            }
            class_neurons
        }).collect();

        let actual_neurons = neurons.len();
        Self { neurons, in_dim: image_dim, n_neurons: actual_neurons }
    }

    /// Competitive forward: all neurons score, return per-class accumulated scores.
    pub fn predict_one(&self, x: &[f32], n_classes: usize) -> Vec<i32> {
        let mut class_scores = vec![0i32; n_classes];
        for neuron in &self.neurons {
            let c = neuron.assigned_class as usize;
            if c < n_classes {
                let s = neuron.score(x);
                if s > 0 {
                    class_scores[c] += s;
                }
            }
        }
        class_scores
    }

    /// Competitive Hebbian refinement (Phase 2) — PARALLEL.
    ///
    /// 1. Score all samples against all neurons (parallel over samples)
    /// 2. Collect reinforce/suppress votes per neuron (parallel reduction)
    /// 3. Apply Hebbian updates (parallel over neurons)
    pub fn refine_step(
        &mut self,
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        _n_classes: usize,
        flip_rate: f32,
    ) {
        let n_neurons = self.neurons.len();

        // Step 1: Parallel scoring — for each sample find best correct + best any
        let votes: Vec<(usize, usize, usize, usize)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let x = &images[i * image_dim..(i + 1) * image_dim];
                let correct_class = labels[i] as usize;

                let mut best_correct_idx = 0usize;
                let mut best_correct_score = i32::MIN;
                let mut best_any_idx = 0usize;
                let mut best_any_score = i32::MIN;

                for (idx, neuron) in self.neurons.iter().enumerate() {
                    let s = neuron.score(x);
                    if neuron.assigned_class as usize == correct_class && s > best_correct_score {
                        best_correct_score = s;
                        best_correct_idx = idx;
                    }
                    if s > best_any_score {
                        best_any_score = s;
                        best_any_idx = idx;
                    }
                }

                // Return: (sample_idx, correct_winner_neuron, any_winner_neuron, correct_class)
                (i, best_correct_idx, best_any_idx, correct_class)
            })
            .collect();

        // Step 2: Apply updates — group by neuron for cache efficiency
        // Collect which samples reinforce/suppress each neuron
        let mut reinforce: Vec<Vec<usize>> = vec![Vec::new(); n_neurons];
        let mut suppress: Vec<Vec<usize>> = vec![Vec::new(); n_neurons];

        for &(sample_idx, correct_winner, any_winner, correct_class) in &votes {
            reinforce[correct_winner].push(sample_idx);
            if self.neurons[any_winner].assigned_class as usize != correct_class {
                suppress[any_winner].push(sample_idx);
            }
        }

        // Step 3: Apply Hebbian updates per neuron (can pick representative sample)
        for neuron_idx in 0..n_neurons {
            // Reinforce with first sample that selected this neuron
            if let Some(&sample_idx) = reinforce[neuron_idx].first() {
                let x = &images[sample_idx * image_dim..(sample_idx + 1) * image_dim];
                self.neurons[neuron_idx].hebbian_update(x, true, flip_rate);
                self.neurons[neuron_idx].wins += reinforce[neuron_idx].len() as u32;
                self.neurons[neuron_idx].correct_wins += reinforce[neuron_idx].len() as u32;
            }
            // Suppress with first sample that mis-selected this neuron
            if let Some(&sample_idx) = suppress[neuron_idx].first() {
                let x = &images[sample_idx * image_dim..(sample_idx + 1) * image_dim];
                self.neurons[neuron_idx].hebbian_update(x, false, flip_rate);
                self.neurons[neuron_idx].wins += suppress[neuron_idx].len() as u32;
            }
        }
    }
}

// ============================================================
// Full Ternary Brain Network
// ============================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TernaryBrain {
    pub layer: CompetitiveLayer,
    pub n_classes: usize,
    pub image_dim: usize,
}

impl TernaryBrain {
    /// Phase 1: Initialize from statistics.
    pub fn init(
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        n_classes: usize,
        neurons_per_class: usize,
    ) -> Self {
        let layer = CompetitiveLayer::from_statistics(
            images, labels, image_dim, n_samples, n_classes, neurons_per_class,
        );
        Self { layer, n_classes, image_dim }
    }

    /// Phase 2: Competitive Hebbian refinement.
    pub fn refine(
        &mut self,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
        n_rounds: usize,
    ) {
        for round in 0..n_rounds {
            // Decreasing flip rate (annealing)
            let flip_rate = 0.05 * (1.0 - round as f32 / n_rounds as f32).max(0.01);
            self.layer.refine_step(
                images, labels, self.image_dim, n_samples, self.n_classes, flip_rate,
            );
        }
    }

    /// Predict batch.
    pub fn predict(&self, images: &[f32], n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let x = &images[i * self.image_dim..(i + 1) * self.image_dim];
                let scores = self.layer.predict_one(x, self.n_classes);
                scores.iter()
                    .enumerate()
                    .max_by_key(|(_, &s)| s)
                    .map(|(c, _)| c as u8)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn accuracy(&self, images: &[f32], labels: &[u8], n_samples: usize) -> f32 {
        let preds = self.predict(images, n_samples);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }

    /// Total ternary weights.
    pub fn total_weights(&self) -> usize {
        self.layer.neurons.iter().map(|n| n.weights.len()).sum()
    }

    /// Flatten all neuron weights into a single f32 vector (for IGQK packing).
    pub fn all_weights_f32(&self) -> Vec<f32> {
        let total = self.total_weights();
        let mut out = Vec::with_capacity(total);
        for neuron in &self.layer.neurons {
            for &w in &neuron.weights {
                out.push(w as f32);
            }
        }
        out
    }

    /// Verify all weights are ternary.
    pub fn verify_ternary(&self) -> bool {
        self.layer.neurons.iter().all(|n| {
            n.weights.iter().all(|&w| w == -1 || w == 0 || w == 1)
        })
    }

    /// Save brain to file (binary bincode format).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, &bytes).map_err(|e| format!("write: {e}"))?;
        Ok(())
    }

    /// Load brain from file.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
        bincode::deserialize(&bytes).map_err(|e| format!("deserialize: {e}"))
    }

    /// Dump all neuron ternary weights as a flat i8 vector (for mutation / storage).
    pub fn dump_weights_i8(&self) -> Vec<i8> {
        let total = self.total_weights();
        let mut out = Vec::with_capacity(total);
        for neuron in &self.layer.neurons {
            for &w in &neuron.weights {
                out.push(w);
            }
        }
        out
    }

    /// Overwrite all neuron weights from a flat ternary i8 vector.
    /// Returns Err if length does not match.
    pub fn load_weights_i8(&mut self, flat: &[i8]) -> Result<(), String> {
        let total = self.total_weights();
        if flat.len() != total {
            return Err(format!(
                "load_weights_i8: expected {} weights, got {}",
                total,
                flat.len()
            ));
        }
        let mut off = 0usize;
        for neuron in self.layer.neurons.iter_mut() {
            let n = neuron.weights.len();
            for k in 0..n {
                let w = flat[off + k];
                neuron.weights[k] = if w > 0 { 1 } else if w < 0 { -1 } else { 0 };
            }
            off += n;
        }
        Ok(())
    }

    /// Clone into a new TernaryBrain from already-loaded template and overwrite its weights.
    /// This is the cheapest way to build a brain for fitness evaluation given a mutated
    /// ternary vector: start from a statistically-initialized template and patch weights.
    pub fn from_template_and_weights(template: &TernaryBrain, flat: &[i8]) -> Result<Self, String> {
        let mut brain = TernaryBrain {
            layer: CompetitiveLayer {
                neurons: template.layer.neurons.clone(),
                in_dim: template.layer.in_dim,
                n_neurons: template.layer.n_neurons,
            },
            n_classes: template.n_classes,
            image_dim: template.image_dim,
        };
        brain.load_weights_i8(flat)?;
        Ok(brain)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;
    use std::time::Instant;

    #[test]
    fn phase1_statistical_init() {
        let data = MnistData::synthetic(2000, 500);

        let start = Instant::now();
        let brain = TernaryBrain::init(
            &data.train_images, &data.train_labels,
            784, data.n_train, 10, 3, // 3 neurons per class = 30 total
        );
        let init_time = start.elapsed();

        let acc = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);

        println!("\n=== Phase 1: Statistical Init ===");
        println!("  Neurons: {} (3 per class)", brain.layer.n_neurons);
        println!("  Weights: {} (100% ternary i8)", brain.total_weights());
        println!("  Init:    {:?}", init_time);
        println!("  Accuracy: {:.1}%", acc * 100.0);
        println!("  Ternary:  {}", if brain.verify_ternary() { "VERIFIED" } else { "FAILED" });

        assert!(brain.verify_ternary());
        assert!(acc > 0.40, "Phase 1 must >40% (got {:.1}%)", acc * 100.0);
    }

    #[test]
    fn phase2_competitive_refinement() {
        let data = MnistData::synthetic(2000, 500);

        let mut brain = TernaryBrain::init(
            &data.train_images, &data.train_labels,
            784, data.n_train, 10, 5, // 5 neurons per class = 50 total
        );

        let acc_before = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);
        println!("\n=== Phase 2: Competitive Hebbian ===");
        println!("  Before refinement: {:.1}%", acc_before * 100.0);

        let start = Instant::now();
        brain.refine(&data.train_images, &data.train_labels, data.n_train, 5);
        let refine_time = start.elapsed();

        let acc_after = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);
        println!("  After 5 rounds:    {:.1}%", acc_after * 100.0);
        println!("  Refinement time:   {:?}", refine_time);
        println!("  Ternary: {}", if brain.verify_ternary() { "VERIFIED" } else { "FAILED" });

        assert!(brain.verify_ternary());
        // Refinement should not destroy accuracy
        assert!(acc_after > acc_before * 0.8,
            "Refinement must not destroy accuracy (before={:.1}%, after={:.1}%)",
            acc_before * 100.0, acc_after * 100.0);
    }

    #[test]
    fn full_brain_training() {
        let data = MnistData::synthetic(2000, 500);

        println!("\n=== Ternary Brain: Full Pipeline ===\n");

        let total_start = Instant::now();

        // Phase 1: Statistical Init
        let mut brain = TernaryBrain::init(
            &data.train_images, &data.train_labels,
            784, data.n_train, 10, 10, // 10 neurons per class = 100 total
        );
        let acc1 = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);
        println!("Phase 1 (init):      {:.1}% | {:?}", acc1 * 100.0, total_start.elapsed());

        // Phase 2: Competitive Refinement (multiple rounds)
        for round in 0..10 {
            brain.refine(&data.train_images, &data.train_labels, data.n_train, 1);
            if round % 3 == 0 || round == 9 {
                let acc = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);
                println!("Phase 2 (round {:>2}):  {:.1}% | {:?}", round + 1, acc * 100.0, total_start.elapsed());
            }
        }

        let total_time = total_start.elapsed();
        let final_acc = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);

        println!("\n  RESULT: {:.1}% accuracy", final_acc * 100.0);
        println!("  Total time: {:?}", total_time);
        println!("  Neurons: {}", brain.layer.n_neurons);
        println!("  Weights: {} (100% ternary i8)", brain.total_weights());
        println!("  Size: {} bytes", brain.total_weights() + brain.layer.n_neurons * 8);
        println!("  Verified ternary: {}", brain.verify_ternary());

        assert!(brain.verify_ternary());
        assert!(final_acc > 0.40,
            "Full brain must >40% (got {:.1}%)", final_acc * 100.0);
    }
}
