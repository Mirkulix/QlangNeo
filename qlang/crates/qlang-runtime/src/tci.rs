//! TCI — Ternary Continual Intelligence.
//!
//! Lifelong learning on top of `TernaryBrain` with:
//! - task snapshots (one ternary weight vector per task)
//! - consensus merge (majority vote over task memories)
//! - task routing via hyperdimensional prototypes
//!
//! Goal: learn multiple tasks sequentially while keeping a compact,
//! interpretable task memory and a global consensus model.

use crate::hdc::HdVector;
use crate::ternary_brain::TernaryBrain;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TaskSnapshot {
    pub name: String,
    pub weights: Vec<i8>,
    pub prototype: HdVector,
    pub train_accuracy: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RouteOutcome {
    pub task_index: usize,
    pub task_name: String,
    pub similarity: f32,
    pub prediction: u8,
    pub confidence: f32,
    pub used_consensus_fallback: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TciEngine {
    pub template: TernaryBrain,
    pub tasks: Vec<TaskSnapshot>,
    pub consensus: Vec<i8>,
}

impl TciEngine {
    /// Create a new continual-learning engine from a template brain.
    pub fn new(template: TernaryBrain) -> Self {
        let consensus = template.dump_weights_i8();
        Self {
            template,
            tasks: Vec::new(),
            consensus,
        }
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn task_names(&self) -> Vec<String> {
        self.tasks.iter().map(|t| t.name.clone()).collect()
    }

    pub fn consensus_weights(&self) -> &[i8] {
        &self.consensus
    }

    /// Persist full TCI state for cross-session continuation.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("tci serialize: {e}"))?;
        std::fs::write(path, &bytes).map_err(|e| format!("tci write: {e}"))
    }

    /// Load full TCI state (template + snapshots + consensus).
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("tci read: {e}"))?;
        bincode::deserialize(&bytes).map_err(|e| format!("tci deserialize: {e}"))
    }

    /// Learn one task and store it as a snapshot.
    ///
    /// The training starts from the current consensus memory so new tasks can
    /// build on previous experience.
    pub fn learn_task(
        &mut self,
        name: &str,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
        n_rounds: usize,
    ) -> Result<f32, String> {
        if n_samples == 0 {
            return Err("learn_task: n_samples must be > 0".into());
        }

        let mut brain = TernaryBrain::from_template_and_weights(&self.template, &self.consensus)?;
        brain.refine(images, labels, n_samples, n_rounds.max(1));

        let eval_n = n_samples.min(2000);
        let train_accuracy = brain.accuracy(images, labels, eval_n);
        let weights = brain.dump_weights_i8();
        let prototype = task_prototype(images, self.template.image_dim, n_samples);

        self.tasks.push(TaskSnapshot {
            name: name.to_string(),
            weights,
            prototype,
            train_accuracy,
        });

        self.consensus = self.build_consensus();
        Ok(train_accuracy)
    }

    /// Route one sample to the most similar task snapshot and predict.
    pub fn route_and_predict(&self, sample: &[f32]) -> Result<RouteOutcome, String> {
        self.route_and_predict_robust(sample, -1.0, 0.0)
    }

    /// Route with safety thresholds.
    ///
    /// Falls back to consensus when route similarity or classifier confidence
    /// are below thresholds.
    pub fn route_and_predict_robust(
        &self,
        sample: &[f32],
        min_similarity: f32,
        min_confidence: f32,
    ) -> Result<RouteOutcome, String> {
        if self.tasks.is_empty() {
            return Err("route_and_predict: no tasks learned".into());
        }
        if sample.len() != self.template.image_dim {
            return Err(format!(
                "route_and_predict: expected sample dim {}, got {}",
                self.template.image_dim,
                sample.len()
            ));
        }

        let sample_sig = HdVector::from_f32(sample);
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;

        for (i, task) in self.tasks.iter().enumerate() {
            let s = sample_sig.similarity(&task.prototype);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }

        let task = &self.tasks[best_i];
        let (prediction, confidence) = self.predict_with_weights(sample, &task.weights)?;

        if best_s < min_similarity || confidence < min_confidence {
            let (cons_pred, cons_conf) = self.predict_with_weights(sample, &self.consensus)?;
            return Ok(RouteOutcome {
                task_index: usize::MAX,
                task_name: "consensus".to_string(),
                similarity: best_s,
                prediction: cons_pred,
                confidence: cons_conf,
                used_consensus_fallback: true,
            });
        }

        Ok(RouteOutcome {
            task_index: best_i,
            task_name: task.name.clone(),
            similarity: best_s,
            prediction,
            confidence,
            used_consensus_fallback: false,
        })
    }

    fn predict_with_weights(&self, sample: &[f32], weights: &[i8]) -> Result<(u8, f32), String> {
        let brain = TernaryBrain::from_template_and_weights(&self.template, weights)?;

        let class_scores = brain.layer.predict_one(sample, brain.n_classes);
        let mut best_cls = 0usize;
        let mut best_score = i32::MIN;
        let mut second_score = i32::MIN;

        for (i, &s) in class_scores.iter().enumerate() {
            if s > best_score {
                second_score = best_score;
                best_score = s;
                best_cls = i;
            } else if s > second_score {
                second_score = s;
            }
        }

        let margin = (best_score - second_score).max(0) as f32;
        let confidence = margin / (best_score.abs().max(1) as f32);
        Ok((best_cls as u8, confidence))
    }

    /// Evaluate one stored task snapshot on any dataset.
    pub fn evaluate_task(
        &self,
        task_index: usize,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
    ) -> Result<f32, String> {
        let task = self
            .tasks
            .get(task_index)
            .ok_or_else(|| format!("evaluate_task: invalid task index {}", task_index))?;
        let brain = TernaryBrain::from_template_and_weights(&self.template, &task.weights)?;
        Ok(brain.accuracy(images, labels, n_samples))
    }

    /// Evaluate the global consensus memory on any dataset.
    pub fn evaluate_consensus(
        &self,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
    ) -> Result<f32, String> {
        let brain = TernaryBrain::from_template_and_weights(&self.template, &self.consensus)?;
        Ok(brain.accuracy(images, labels, n_samples))
    }

    fn build_consensus(&self) -> Vec<i8> {
        if self.tasks.is_empty() {
            return self.template.dump_weights_i8();
        }

        let base = self.template.dump_weights_i8();
        let n = base.len();
        let mut out = vec![0i8; n];

        for i in 0..n {
            let mut vote = 0i32;
            for task in &self.tasks {
                vote += task.weights[i] as i32;
            }
            out[i] = if vote > 0 {
                1
            } else if vote < 0 {
                -1
            } else {
                base[i]
            };
        }
        out
    }
}

fn task_prototype(images: &[f32], image_dim: usize, n_samples: usize) -> HdVector {
    let mut mean = vec![0.0f32; image_dim];
    let n = n_samples.max(1);

    for s in 0..n_samples {
        let off = s * image_dim;
        for k in 0..image_dim {
            mean[k] += images[off + k];
        }
    }
    for v in &mut mean {
        *v /= n as f32;
    }

    HdVector::from_f32(&mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;

    #[test]
    fn learns_two_tasks_and_routes() {
        let data = MnistData::synthetic(1200, 200);
        let image_dim = 784;

        let template = TernaryBrain::init(
            &data.train_images,
            &data.train_labels,
            image_dim,
            600,
            10,
            8,
        );

        let mut engine = TciEngine::new(template);

        let n_a = 500usize;
        let acc_a = engine
            .learn_task(
                "digits-normal",
                &data.train_images[..n_a * image_dim],
                &data.train_labels[..n_a],
                n_a,
                3,
            )
            .unwrap();
        assert!(acc_a >= 0.0);

        let n_b = 500usize;
        let mut inv_images = data.train_images[..n_b * image_dim].to_vec();
        for v in &mut inv_images {
            *v = 1.0 - *v;
        }
        let acc_b = engine
            .learn_task(
                "digits-inverted",
                &inv_images,
                &data.train_labels[..n_b],
                n_b,
                3,
            )
            .unwrap();
        assert!(acc_b >= 0.0);

        assert_eq!(engine.task_count(), 2);
        assert_eq!(engine.consensus_weights().len(), engine.template.total_weights());

        let sample = &data.test_images[..image_dim];
        let routed = engine.route_and_predict(sample).unwrap();
        assert!(routed.task_index < 2);
        assert!(!routed.used_consensus_fallback);
    }

    #[test]
    fn persistence_roundtrip() {
        let data = MnistData::synthetic(800, 100);
        let image_dim = 784;

        let template = TernaryBrain::init(
            &data.train_images,
            &data.train_labels,
            image_dim,
            500,
            10,
            6,
        );

        let mut engine = TciEngine::new(template);
        let _ = engine
            .learn_task(
                "task-a",
                &data.train_images[..400 * image_dim],
                &data.train_labels[..400],
                400,
                2,
            )
            .unwrap();

        let path = "/tmp/tci_roundtrip.qltc";
        engine.save(path).unwrap();
        let loaded = TciEngine::load(path).unwrap();

        assert_eq!(loaded.task_count(), 1);
        assert_eq!(loaded.task_names()[0], "task-a");
        assert_eq!(loaded.consensus_weights().len(), engine.consensus_weights().len());
    }

    #[test]
    fn robust_routing_can_fallback() {
        let data = MnistData::synthetic(1000, 100);
        let image_dim = 784;

        let template = TernaryBrain::init(
            &data.train_images,
            &data.train_labels,
            image_dim,
            600,
            10,
            6,
        );

        let mut engine = TciEngine::new(template);
        let _ = engine
            .learn_task(
                "task-a",
                &data.train_images[..500 * image_dim],
                &data.train_labels[..500],
                500,
                2,
            )
            .unwrap();

        let noisy = vec![0.5f32; image_dim];
        let routed = engine
            .route_and_predict_robust(&noisy, 0.95, 0.95)
            .unwrap();
        assert!(routed.used_consensus_fallback);
        assert_eq!(routed.task_name, "consensus");
    }

    #[test]
    fn snapshot_retention_after_more_tasks() {
        let data = MnistData::synthetic(1600, 200);
        let image_dim = 784;

        let template = TernaryBrain::init(
            &data.train_images,
            &data.train_labels,
            image_dim,
            900,
            10,
            8,
        );

        let mut engine = TciEngine::new(template);

        let n = 500usize;
        let _ = engine
            .learn_task(
                "normal",
                &data.train_images[..n * image_dim],
                &data.train_labels[..n],
                n,
                3,
            )
            .unwrap();

        let eval_n = 150usize;
        let before = engine
            .evaluate_task(0, &data.test_images[..eval_n * image_dim], &data.test_labels[..eval_n], eval_n)
            .unwrap();

        let mut inv = data.train_images[n * image_dim..2 * n * image_dim].to_vec();
        for v in &mut inv {
            *v = 1.0 - *v;
        }
        let _ = engine
            .learn_task("inverted", &inv, &data.train_labels[n..2 * n], n, 3)
            .unwrap();

        let mut contrast = data.train_images[2 * n * image_dim..3 * n * image_dim].to_vec();
        for v in &mut contrast {
            *v = (*v * 1.4).clamp(0.0, 1.0);
        }
        let _ = engine
            .learn_task("contrast", &contrast, &data.train_labels[2 * n..3 * n], n, 3)
            .unwrap();

        let after = engine
            .evaluate_task(0, &data.test_images[..eval_n * image_dim], &data.test_labels[..eval_n], eval_n)
            .unwrap();

        // Snapshot retention should hold strongly despite learning new tasks.
        assert!((before - after).abs() <= 0.02);
    }
}
