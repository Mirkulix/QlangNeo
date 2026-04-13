//! TCI + Evolution Daemon Integration
//!
//! Couples Ternary Continual Intelligence (TCI) with population-based evolution.
//! Manages specialist populations for each task, with evolution improving solutions
//! that feed back into the TCI consensus.
//!
//! Architecture:
//! 1. TciEngine maintains task snapshots + global consensus weights.
//! 2. For each learned task, spawn a specialist population that evolves on that task.
//! 3. Top-fitness specialists' weights are merged back into the task snapshot.
//! 4. New tasks learn from the updated consensus.
//! 5. Specialists share entanglement info via HDC prototypes for cross-task learning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::ternary_brain::TernaryBrain;
use crate::tci::{RouteOutcome, TciEngine};

use super::mutation::{mutate_ternary_i8, MutationConfig};

/// Per-task specialist population snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpecialistPool {
    /// Task name (links to TciEngine task snapshot).
    pub task_name: String,
    /// Active specialists: each is a mutated variant on the task's weights.
    pub specialists: Vec<SpecialistBrain>,
    /// Fitness scores for each specialist (parallel to `specialists`).
    pub fitness_scores: Vec<f32>,
    /// Generation counter for this task's evolution.
    pub generation: u32,
    /// Total mutations spawned for this task.
    pub total_mutations: u32,
}

/// A single specialist — the task's weights mutated + adapted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistBrain {
    pub id: String,
    pub brain: TernaryBrain,
    pub generation_born: u32,
    pub parent_id: Option<String>,
    pub mutations_applied: u32,
}

/// Configuration for TCI + evolution coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TciEvolutionConfig {
    /// Target population size per task.
    pub pool_size_per_task: usize,
    /// Mutations spawned per generation per task.
    pub mutations_per_gen: usize,
    /// Per-weight flip probability in mutations.
    pub mutation_rate: f32,
    /// Fraction of low-fitness specialists retired each generation.
    pub retire_fraction: f32,
    /// How many top specialists to merge back into task consensus.
    pub merge_top_k: usize,
    /// RNG seed for deterministic reproduction.
    pub seed: u64,
}

impl Default for TciEvolutionConfig {
    fn default() -> Self {
        Self {
            pool_size_per_task: 20,
            mutations_per_gen: 5,
            mutation_rate: 0.02,
            retire_fraction: 0.3,
            merge_top_k: 3,
            seed: 42,
        }
    }
}

/// Result of one evolution generation tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionGenerationReport {
    pub task_name: String,
    pub generation: u32,
    pub population_size: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
    pub mutations_spawned: usize,
    pub retired: usize,
    pub merged_to_consensus: usize,
    pub elapsed_ms: u64,
}

/// TCI + Evolution daemon: orchestrates task-level specialist populations.
pub struct TciEvolutionDaemon {
    config: Mutex<TciEvolutionConfig>,
    engine: Mutex<TciEngine>,
    pools: Mutex<HashMap<String, TaskSpecialistPool>>,
    reports: Mutex<Vec<EvolutionGenerationReport>>,
    running: AtomicBool,
    total_generations: AtomicU32,
}

impl TciEvolutionDaemon {
    /// Create a new TCI+Evolution daemon with fresh engine.
    /// The engine needs a template brain for initialization.
    pub fn with_template(template: TernaryBrain) -> Arc<Self> {
        Arc::new(Self {
            config: Mutex::new(TciEvolutionConfig::default()),
            engine: Mutex::new(TciEngine::new(template)),
            pools: Mutex::new(HashMap::new()),
            reports: Mutex::new(Vec::new()),
            running: AtomicBool::new(false),
            total_generations: AtomicU32::new(0),
        })
    }

    /// Create from an existing TciEngine (for recovery / continuation).
    pub fn from_engine(engine: TciEngine) -> Arc<Self> {
        Arc::new(Self {
            config: Mutex::new(TciEvolutionConfig::default()),
            engine: Mutex::new(engine),
            pools: Mutex::new(HashMap::new()),
            reports: Mutex::new(Vec::new()),
            running: AtomicBool::new(false),
            total_generations: AtomicU32::new(0),
        })
    }

    /// Learn a new task using naive TCI, then spawn specialist population.
    pub fn learn_task_with_evolution(
        &self,
        task_name: &str,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
        learning_rounds: usize,
        evolution_rounds: usize,
    ) -> Result<TaskSpecialistPool, String> {
        // Step 1: Learn task via TCI baseline.
        {
            let mut engine = self.engine.lock().unwrap();
            engine
                .learn_task(task_name, images, labels, n_samples, learning_rounds)
                .map_err(|e| format!("TCI learn_task failed: {}", e))?;
        }

        // Step 2: Spawn initial specialist population from the task's brain.
        let cfg = self.config.lock().unwrap().clone();
        let base_brain = {
            let engine = self.engine.lock().unwrap();
            engine.template.clone()
        };

        let _rng = super::mutation::XorShift64::new(cfg.seed);
        let mut specialists = Vec::new();
        for i in 0..cfg.pool_size_per_task {
            let id = format!("{}-spec-{:06x}", task_name, i);
            specialists.push(SpecialistBrain {
                id,
                brain: base_brain.clone(),
                generation_born: 0,
                parent_id: None,
                mutations_applied: 0,
            });
        }

        let mut pool = TaskSpecialistPool {
            task_name: task_name.to_string(),
            specialists,
            fitness_scores: vec![0.0; cfg.pool_size_per_task],
            generation: 0,
            total_mutations: 0,
        };

        // Step 3: Evaluate initial population.
        self.evaluate_pool(&mut pool, images, labels, n_samples)?;

        // Step 4: Evolve for N generations.
        for _ in 0..evolution_rounds {
            let tick_start = Instant::now();
            let mut report = self.evolve_generation(&mut pool, images, labels, n_samples)?;
            let elapsed_ms = tick_start.elapsed().as_millis() as u64;
            report.elapsed_ms = elapsed_ms;

            self.reports.lock().unwrap().push(report);
            self.total_generations.fetch_add(1, Ordering::Relaxed);
        }

        // Step 5: Merge top specialists back into TCI task snapshot.
        self.merge_best_into_task(&pool, task_name)?;

        self.pools
            .lock()
            .unwrap()
            .insert(task_name.to_string(), pool.clone());

        Ok(pool)
    }

    /// Evaluate all specialists in a pool on the given dataset.
    fn evaluate_pool(
        &self,
        pool: &mut TaskSpecialistPool,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
    ) -> Result<(), String> {
        for (i, specialist) in pool.specialists.iter().enumerate() {
            let mut correct = 0;
            let preds = specialist.brain.predict(images, n_samples);
            for (j, &pred) in preds.iter().enumerate() {
                if let Some(&label) = labels.get(j) {
                    if pred == label {
                        correct += 1;
                    }
                }
            }
            pool.fitness_scores[i] = (correct as f32) / (n_samples.min(preds.len()) as f32);
        }

        Ok(())
    }

    /// Evolve the pool by one generation: mutation, evaluation, selection.
    fn evolve_generation(
        &self,
        pool: &mut TaskSpecialistPool,
        images: &[f32],
        labels: &[u8],
        n_samples: usize,
    ) -> Result<EvolutionGenerationReport, String> {
        let cfg = self.config.lock().unwrap().clone();

        // Retirement: kill low-fitness tail.
        let retire_count = ((pool.specialists.len() as f32) * cfg.retire_fraction).max(1.0) as usize;
        let mut indices: Vec<usize> = (0..pool.fitness_scores.len()).collect();
        indices.sort_by(|&a, &b| {
            pool.fitness_scores[b]
                .partial_cmp(&pool.fitness_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top (size - retire_count), remove rest.
        let keep_count = (pool.specialists.len() - retire_count).max(1);
        pool.specialists.truncate(keep_count);
        pool.fitness_scores.truncate(keep_count);

        let retired = retire_count;

        // Mutation: spawn offspring from top performers.
        let mut mutation_config = MutationConfig::default();
        mutation_config.flip_rate = cfg.mutation_rate;
        mutation_config.seed = cfg.seed ^ pool.generation as u64;

        let top_performers: Vec<usize> = (0..pool.specialists.len()).collect();
        for _ in std::iter::repeat(()).take(cfg.mutations_per_gen) {
            let parent_idx = if !top_performers.is_empty() {
                top_performers[cfg.seed as usize % top_performers.len()]
            } else {
                0
            };
            
            if parent_idx < pool.specialists.len() {
                let parent = &pool.specialists[parent_idx];

                let mut child_weights = parent.brain.dump_weights_i8();
                mutate_ternary_i8(&mut child_weights, &mutation_config);

                let mut child_brain = parent.brain.clone();
                child_brain.load_weights_i8(&child_weights)?;

                let child_id = format!("{}-gen{}-mut{}", pool.task_name, pool.generation, parent_idx);
                let child = SpecialistBrain {
                    id: child_id,
                    brain: child_brain,
                    generation_born: pool.generation + 1,
                    parent_id: Some(parent.id.clone()),
                    mutations_applied: parent.mutations_applied + 1,
                };

                pool.specialists.push(child);
                pool.fitness_scores.push(0.0);
                pool.total_mutations += 1;
            }
        }

        // Evaluate new offspring (and existing).
        self.evaluate_pool(pool, images, labels, n_samples)?;

        // Get stats.
        let best_fitness = pool
            .fitness_scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let avg_fitness = pool.fitness_scores.iter().sum::<f32>() / pool.fitness_scores.len().max(1) as f32;
        let worst_fitness = pool
            .fitness_scores
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        pool.generation += 1;

        Ok(EvolutionGenerationReport {
            task_name: pool.task_name.clone(),
            generation: pool.generation,
            population_size: pool.specialists.len(),
            best_fitness,
            avg_fitness,
            worst_fitness,
            mutations_spawned: cfg.mutations_per_gen,
            retired,
            merged_to_consensus: 0, // Will update after merge_best_into_task.
            elapsed_ms: 0, // Caller fills in.
        })
    }

    /// Merge top-K specialists' weights back into the task snapshot.
    fn merge_best_into_task(&self, pool: &TaskSpecialistPool, task_name: &str) -> Result<(), String> {
        let cfg = self.config.lock().unwrap().clone();
        let mut engine = self.engine.lock().unwrap();

        if pool.specialists.is_empty() {
            return Ok(());
        }

        // Find top K by fitness.
        let mut indices: Vec<usize> = (0..pool.specialists.len()).collect();
        indices.sort_by(|&a, &b| {
            pool.fitness_scores[b]
                .partial_cmp(&pool.fitness_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let merge_k = cfg.merge_top_k.min(indices.len());
        let max_weight_count = pool.specialists[0].brain.dump_weights_i8().len();

        // Merge top K weights via majority vote (similar to TCI consensus).
        let mut merged_weights = vec![0i8; max_weight_count];
        
        for (w_idx, w) in merged_weights.iter_mut().enumerate() {
            let mut votes = HashMap::new();
            for &spec_idx in indices.iter().take(merge_k) {
                let specialist_weights = pool.specialists[spec_idx].brain.dump_weights_i8();
                if let Some(&weight) = specialist_weights.get(w_idx) {
                    *votes.entry(weight).or_insert(0usize) += 1;
                }
            }
            if let Some((&final_w, _)) = votes.iter().max_by_key(|(_, &count)| count) {
                *w = final_w;
            }
        }

        // Update TCI task snapshot with merged weights (if task exists).
        if let Some(task_idx) = engine.tasks.iter().position(|t| t.name == task_name) {
            engine.tasks[task_idx].weights = merged_weights.clone();
            // Re-build consensus by majority vote over all task snapshots.
            let mut consensus = vec![0i8; max_weight_count];
            for (w_idx, w) in consensus.iter_mut().enumerate() {
                let mut votes = HashMap::new();
                for task in &engine.tasks {
                    if let Some(&weight) = task.weights.get(w_idx) {
                        *votes.entry(weight).or_insert(0usize) += 1;
                    }
                }
                if let Some((&final_w, _)) = votes.iter().max_by_key(|(_, &count)| count) {
                    *w = final_w;
                }
            }
            engine.consensus = consensus;
        }

        Ok(())
    }

    /// Save the entire daemon state (engine + pools + config).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let state = DaemonState {
            config: self.config.lock().unwrap().clone(),
            engine: self.engine.lock().unwrap().clone(),
            pools: self.pools.lock().unwrap().clone(),
            reports: self.reports.lock().unwrap().clone(),
        };

        let serialized =
            bincode::serialize(&state).map_err(|e| format!("Serialization failed: {}", e))?;
        std::fs::write(path, serialized).map_err(|e| format!("Write failed: {}", e))?;
        Ok(())
    }

    /// Load daemon state from disk.
    pub fn load(path: &str) -> Result<Arc<Self>, String> {
        let data = std::fs::read(path).map_err(|e| format!("Read failed: {}", e))?;
        let state: DaemonState =
            bincode::deserialize(&data).map_err(|e| format!("Deserialization failed: {}", e))?;

        Ok(Arc::new(Self {
            config: Mutex::new(state.config),
            engine: Mutex::new(state.engine),
            pools: Mutex::new(state.pools),
            reports: Mutex::new(state.reports),
            running: AtomicBool::new(false),
            total_generations: AtomicU32::new(0),
        }))
    }

    /// Get current status.
    pub fn status(&self) -> DaemonStatus {
        let pools = self.pools.lock().unwrap();
        DaemonStatus {
            running: self.running.load(Ordering::Relaxed),
            total_tasks: pools.len(),
            total_specialists: pools.values().map(|p| p.specialists.len()).sum(),
            total_generations: self.total_generations.load(Ordering::Relaxed),
            tasks: pools
                .values()
                .map(|p| TaskStatus {
                    name: p.task_name.clone(),
                    population_size: p.specialists.len(),
                    generation: p.generation,
                    best_fitness: p
                        .fitness_scores
                        .iter()
                        .copied()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0.0),
                    avg_fitness: p.fitness_scores.iter().sum::<f32>() / p.fitness_scores.len().max(1) as f32,
                })
                .collect(),
        }
    }

    /// Get evolution reports for a task.
    pub fn task_reports(&self, task_name: &str) -> Vec<EvolutionGenerationReport> {
        self.reports
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.task_name == task_name)
            .cloned()
            .collect()
    }

    /// Query a sample using TCI routing (internal engine).
    pub fn route_sample(&self, sample: &[f32]) -> Result<RouteOutcome, String> {
        self.engine
            .lock()
            .unwrap()
            .route_and_predict_robust(sample, 0.5, 0.5)
            .map_err(|e| format!("Routing failed: {}", e))
    }
}

/// Serializable daemon state (for save/load).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DaemonState {
    config: TciEvolutionConfig,
    engine: TciEngine,
    pools: HashMap<String, TaskSpecialistPool>,
    reports: Vec<EvolutionGenerationReport>,
}

/// Status snapshot for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub total_tasks: usize,
    pub total_specialists: usize,
    pub total_generations: u32,
    pub tasks: Vec<TaskStatus>,
}

/// Per-task status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub name: String,
    pub population_size: usize,
    pub generation: u32,
    pub best_fitness: f32,
    pub avg_fitness: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_save_load() -> Result<(), String> {
        // Create a minimal template brain
        let template = TernaryBrain::init(
            &vec![0.0; 784 * 10],
            &vec![0; 10],
            784,
            10,
            10,
            10,
        );

        let daemon = TciEvolutionDaemon::with_template(template);
        let path = "/tmp/tci_evolution_test.qltc";

        daemon.save(path)?;
        let _reloaded = TciEvolutionDaemon::load(path)?;

        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
