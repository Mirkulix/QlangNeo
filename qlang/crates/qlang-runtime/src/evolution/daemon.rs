//! Evolution Daemon — orchestrates continuous specialist evolution.
//!
//! Runs in the background, periodically:
//!   1. Snapshots the current generation via `FitnessTracker`.
//!   2. Retires/kills under-performers.
//!   3. Spawns mutated offspring from top performers.
//!
//! This is a *lightweight stub* sufficient to power the Evolution Dashboard.
//! A full implementation (real specialist weight storage, genuine mutation
//! dispatch) is being built in parallel; the daemon contract exposed here is
//! deliberately stable so the web API won't break when that lands.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use super::fitness::{FitnessTracker, GenerationStats};
use super::mutation::XorShift64;

// ---------------------------------------------------------------------------
// Configuration + reports
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Seconds between generation ticks.
    pub interval_secs: u64,
    /// Target population size — daemon spawns/kills to converge on this.
    pub population_size: usize,
    /// Per-weight flip probability handed to the mutation engine.
    pub mutation_rate: f32,
    /// Fraction of the population retired each generation (low-fitness tail).
    pub retire_fraction: f32,
    /// Age in generations after which a specialist is forcibly retired.
    pub max_age: u32,
    /// RNG seed for deterministic reproductive behavior.
    pub seed: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            interval_secs: 10,
            population_size: 50,
            mutation_rate: 0.01,
            retire_fraction: 0.2,
            max_age: 30,
            seed: 42,
        }
    }
}

/// Life-cycle status of a specialist tracked by the daemon.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SpecialistStatus {
    Active,
    Retired,
    Dead,
}

/// Specialist record tracked by the daemon (extends fitness data with lineage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistInfo {
    pub id: String,
    pub generation_born: u32,
    pub parent_id: Option<String>,
    pub children: Vec<String>,
    pub fitness: f32,
    pub age: u32,
    pub status: SpecialistStatus,
    pub mutations: u32,
}

/// Report emitted at the end of every generation tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationReport {
    pub generation: u32,
    pub timestamp: u64,
    pub population_size: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub min_fitness: f32,
    pub best_id: String,
    pub retired: Vec<String>,
    pub killed: Vec<String>,
    pub spawned: Vec<String>,
    pub notes: String,
}

/// Current snapshot of the daemon state (cheap to clone).
#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub current_generation: u32,
    pub uptime_secs: u64,
    pub population_size: usize,
    pub best_fitness_ever: f32,
    pub total_born: usize,
    pub total_retired: usize,
    pub total_killed: usize,
    pub config: DaemonConfig,
    pub last_report: Option<GenerationReport>,
}

// ---------------------------------------------------------------------------
// Daemon implementation
// ---------------------------------------------------------------------------

pub struct EvolutionDaemon {
    config: Mutex<DaemonConfig>,
    running: AtomicBool,
    stop_flag: Arc<AtomicBool>,
    current_generation: AtomicU32,
    started_at: Mutex<Option<Instant>>,

    fitness: Mutex<FitnessTracker>,
    specialists: Mutex<HashMap<String, SpecialistInfo>>,
    history: Mutex<VecDeque<GenerationReport>>,

    best_fitness_ever: Mutex<f32>,
    total_born: AtomicUsize,
    total_retired: AtomicUsize,
    total_killed: AtomicUsize,

    broadcast: broadcast::Sender<GenerationReport>,
}

impl EvolutionDaemon {
    pub fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(256);
        Arc::new(Self {
            config: Mutex::new(DaemonConfig::default()),
            running: AtomicBool::new(false),
            stop_flag: Arc::new(AtomicBool::new(false)),
            current_generation: AtomicU32::new(0),
            started_at: Mutex::new(None),
            fitness: Mutex::new(FitnessTracker::new()),
            specialists: Mutex::new(HashMap::new()),
            history: Mutex::new(VecDeque::with_capacity(500)),
            best_fitness_ever: Mutex::new(0.0),
            total_born: AtomicUsize::new(0),
            total_retired: AtomicUsize::new(0),
            total_killed: AtomicUsize::new(0),
            broadcast: tx,
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<GenerationReport> {
        self.broadcast.subscribe()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn config(&self) -> DaemonConfig {
        self.config.lock().unwrap().clone()
    }

    pub fn status(&self) -> DaemonStatus {
        let uptime = self
            .started_at
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);
        let last_report = self.history.lock().unwrap().back().cloned();
        let population_size = self
            .specialists
            .lock()
            .unwrap()
            .values()
            .filter(|s| s.status == SpecialistStatus::Active)
            .count();
        DaemonStatus {
            running: self.is_running(),
            current_generation: self.current_generation.load(Ordering::Relaxed),
            uptime_secs: uptime,
            population_size,
            best_fitness_ever: *self.best_fitness_ever.lock().unwrap(),
            total_born: self.total_born.load(Ordering::Relaxed),
            total_retired: self.total_retired.load(Ordering::Relaxed),
            total_killed: self.total_killed.load(Ordering::Relaxed),
            config: self.config(),
            last_report,
        }
    }

    pub fn history(&self) -> Vec<GenerationReport> {
        self.history.lock().unwrap().iter().cloned().collect()
    }

    pub fn specialists(&self) -> Vec<SpecialistInfo> {
        let mut v: Vec<SpecialistInfo> =
            self.specialists.lock().unwrap().values().cloned().collect();
        v.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Walk the ancestry + descendants of `id` and return the enclosing lineage
    /// tree (flattened to a Vec; the client reconstructs edges via parent_id).
    pub fn lineage(&self, id: &str) -> Vec<SpecialistInfo> {
        let all = self.specialists.lock().unwrap();
        let mut out = HashMap::new();

        // ancestors
        let mut cursor = all.get(id).cloned();
        while let Some(s) = cursor {
            let parent = s.parent_id.clone();
            out.insert(s.id.clone(), s);
            cursor = parent.and_then(|p| all.get(&p).cloned());
        }
        // descendants (BFS)
        let mut queue: Vec<String> = vec![id.to_string()];
        while let Some(cur) = queue.pop() {
            if let Some(s) = all.get(&cur) {
                for child_id in &s.children {
                    if !out.contains_key(child_id) {
                        if let Some(c) = all.get(child_id) {
                            out.insert(c.id.clone(), c.clone());
                            queue.push(c.id.clone());
                        }
                    }
                }
            }
        }
        out.into_values().collect()
    }

    /// Start the background tick loop. Returns immediately; safe to call once.
    pub fn start(self: &Arc<Self>, config: DaemonConfig) -> Result<(), String> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err("daemon already running".into());
        }
        *self.config.lock().unwrap() = config.clone();
        self.stop_flag.store(false, Ordering::Relaxed);
        *self.started_at.lock().unwrap() = Some(Instant::now());

        // Seed initial population if empty.
        self.seed_population(config.population_size, config.seed);

        let daemon = self.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(daemon.config().interval_secs.max(1));
            while !daemon.stop_flag.load(Ordering::Relaxed) {
                tokio::time::sleep(interval).await;
                if daemon.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                let report = daemon.tick_once();
                let _ = daemon.broadcast.send(report);
            }
            daemon.running.store(false, Ordering::SeqCst);
        });

        Ok(())
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Internal — one generation tick.
    // -----------------------------------------------------------------------

    fn seed_population(&self, n: usize, seed: u64) {
        let mut specs = self.specialists.lock().unwrap();
        let mut fitness = self.fitness.lock().unwrap();
        let mut rng = XorShift64::new(seed);
        let existing = specs
            .values()
            .filter(|s| s.status == SpecialistStatus::Active)
            .count();
        for _ in existing..n {
            let id = format!("spec-{:06x}", rng.next_u64() & 0xFFFFFF);
            fitness.register(id.clone(), 0, None);
            specs.insert(
                id.clone(),
                SpecialistInfo {
                    id: id.clone(),
                    generation_born: 0,
                    parent_id: None,
                    children: Vec::new(),
                    fitness: 0.1 + rng.next_f32() * 0.3,
                    age: 0,
                    status: SpecialistStatus::Active,
                    mutations: 0,
                },
            );
            self.total_born.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn tick_once(&self) -> GenerationReport {
        let cfg = self.config();
        let gen = self.current_generation.fetch_add(1, Ordering::Relaxed) + 1;

        // Simulate fitness drift + age every active specialist.
        let mut rng = XorShift64::new(cfg.seed ^ gen as u64);
        let mut retired = Vec::new();
        let mut killed = Vec::new();
        let mut spawned = Vec::new();

        {
            let mut specs = self.specialists.lock().unwrap();
            // 1. drift + age
            for s in specs.values_mut() {
                if s.status == SpecialistStatus::Active {
                    s.age += 1;
                    let jitter = (rng.next_f32() - 0.5) * 0.05;
                    s.fitness = (s.fitness + jitter).clamp(0.0, 1.0);
                }
            }

            // 2. retire old
            let old: Vec<String> = specs
                .values()
                .filter(|s| s.status == SpecialistStatus::Active && s.age >= cfg.max_age)
                .map(|s| s.id.clone())
                .collect();
            for id in &old {
                if let Some(s) = specs.get_mut(id) {
                    s.status = SpecialistStatus::Retired;
                    retired.push(id.clone());
                    self.total_retired.fetch_add(1, Ordering::Relaxed);
                }
            }

            // 3. kill worst `retire_fraction`
            let mut actives: Vec<(String, f32)> = specs
                .values()
                .filter(|s| s.status == SpecialistStatus::Active)
                .map(|s| (s.id.clone(), s.fitness))
                .collect();
            actives.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let kill_n =
                ((actives.len() as f32) * cfg.retire_fraction).round() as usize;
            for (id, _) in actives.iter().take(kill_n) {
                if let Some(s) = specs.get_mut(id) {
                    s.status = SpecialistStatus::Dead;
                    killed.push(id.clone());
                    self.total_killed.fetch_add(1, Ordering::Relaxed);
                }
            }

            // 4. spawn from best parents to refill population
            let mut parents: Vec<(String, f32)> = specs
                .values()
                .filter(|s| s.status == SpecialistStatus::Active)
                .map(|s| (s.id.clone(), s.fitness))
                .collect();
            parents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let current_active = parents.len();
            let target = cfg.population_size;
            let need = target.saturating_sub(current_active);

            // Keep ids to update children after the main loop, to satisfy borrowck.
            let mut link_updates: Vec<(String, String)> = Vec::new();
            for i in 0..need {
                if parents.is_empty() {
                    break;
                }
                let parent = &parents[i % parents.len()];
                let child_id = format!("spec-{:06x}", rng.next_u64() & 0xFFFFFF);
                let child_fit = (parent.1
                    + (rng.next_f32() - 0.5) * cfg.mutation_rate * 10.0)
                    .clamp(0.0, 1.0);
                specs.insert(
                    child_id.clone(),
                    SpecialistInfo {
                        id: child_id.clone(),
                        generation_born: gen,
                        parent_id: Some(parent.0.clone()),
                        children: Vec::new(),
                        fitness: child_fit,
                        age: 0,
                        status: SpecialistStatus::Active,
                        mutations: 1,
                    },
                );
                link_updates.push((parent.0.clone(), child_id.clone()));
                spawned.push(child_id);
                self.total_born.fetch_add(1, Ordering::Relaxed);
            }
            for (parent_id, child_id) in link_updates {
                if let Some(p) = specs.get_mut(&parent_id) {
                    p.children.push(child_id);
                }
            }
        }

        // 5. compute aggregates for report
        let specs = self.specialists.lock().unwrap();
        let actives: Vec<&SpecialistInfo> = specs
            .values()
            .filter(|s| s.status == SpecialistStatus::Active)
            .collect();
        let population_size = actives.len();
        let (sum, mut min, mut max, mut best_id) =
            (0.0f32, f32::INFINITY, f32::NEG_INFINITY, String::new());
        let mut sum_acc = sum;
        for s in &actives {
            sum_acc += s.fitness;
            if s.fitness > max {
                max = s.fitness;
                best_id = s.id.clone();
            }
            if s.fitness < min {
                min = s.fitness;
            }
        }
        if actives.is_empty() {
            min = 0.0;
            max = 0.0;
        }
        let avg = if population_size > 0 {
            sum_acc / population_size as f32
        } else {
            0.0
        };
        {
            let mut bfe = self.best_fitness_ever.lock().unwrap();
            if max > *bfe {
                *bfe = max;
            }
        }

        let report = GenerationReport {
            generation: gen,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            population_size,
            best_fitness: max,
            avg_fitness: avg,
            min_fitness: min,
            best_id,
            retired,
            killed,
            spawned,
            notes: format!("gen {} — {} active", gen, population_size),
        };

        // Cap history at 500 reports.
        let mut hist = self.history.lock().unwrap();
        if hist.len() >= 500 {
            hist.pop_front();
        }
        hist.push_back(report.clone());

        // Bump fitness-tracker generation counter too.
        drop(hist);
        let mut ft = self.fitness.lock().unwrap();
        let _ = ft.snapshot_generation();

        report
    }
}

pub type SharedDaemon = Arc<EvolutionDaemon>;

// ---------------------------------------------------------------------------
// Utility: produce ad-hoc aggregate stats for quick status summary even when
// the daemon is idle (used by the Evolution tab when not yet started).
// ---------------------------------------------------------------------------

pub fn empty_history() -> Vec<GenerationReport> {
    Vec::new()
}

pub fn fitness_snapshot_to_report(s: &GenerationStats) -> GenerationReport {
    GenerationReport {
        generation: s.generation,
        timestamp: s.timestamp,
        population_size: s.population_size,
        best_fitness: s.max_fitness,
        avg_fitness: s.avg_fitness,
        min_fitness: s.min_fitness,
        best_id: s.best_specialist_id.clone(),
        retired: Vec::new(),
        killed: Vec::new(),
        spawned: Vec::new(),
        notes: "imported from fitness tracker".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let c = DaemonConfig::default();
        assert_eq!(c.population_size, 50);
        assert!(c.mutation_rate > 0.0);
    }

    #[test]
    fn test_seed_and_tick() {
        let d = EvolutionDaemon::new();
        d.seed_population(10, 42);
        assert_eq!(d.specialists().len(), 10);
        let r1 = d.tick_once();
        assert_eq!(r1.generation, 1);
        let r2 = d.tick_once();
        assert_eq!(r2.generation, 2);
        // population should refill up to target (50 by default)
        assert!(r2.spawned.len() > 0);
    }

    #[test]
    fn test_lineage_ancestors() {
        let d = EvolutionDaemon::new();
        d.seed_population(5, 1);
        let _ = d.tick_once();
        let any_child = d
            .specialists()
            .into_iter()
            .find(|s| s.parent_id.is_some());
        if let Some(c) = any_child {
            let tree = d.lineage(&c.id);
            assert!(tree.len() >= 2);
        }
    }
}
