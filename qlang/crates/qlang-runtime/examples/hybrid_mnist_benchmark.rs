// Hybrid Spiking Classifier MNIST Benchmark
//! 
//! Trains HybridSpikingClassifier on synthetic MNIST-like data.
//! Goal: Achieve >=90% accuracy to prove the architecture works.
//!
//! Build: `cargo build --release --example hybrid_mnist_benchmark`
//! Run: `cargo run --release --example hybrid_mnist_benchmark --manifest-path crates/qlang-runtime/Cargo.toml`

use qlang_runtime::hybrid_spiking::HybridSpikingClassifier;
use std::env;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum BenchProfile {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug)]
struct BenchConfig {
    profile: BenchProfile,
    timesteps: usize,
    n_train_samples: usize,
    n_test_samples: usize,
    epochs: usize,
    eval_every: usize,
    use_feature_cache: bool,
}

fn parse_profile() -> BenchConfig {
    let args: Vec<String> = env::args().collect();
    let mut profile = BenchProfile::Fast;
    let mut use_feature_cache = true;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" if i + 1 < args.len() => {
                profile = match args[i + 1].as_str() {
                    "full" => BenchProfile::Full,
                    "balanced" => BenchProfile::Balanced,
                    _ => BenchProfile::Fast,
                };
                i += 1;
            }
            "--no-cache" => {
                use_feature_cache = false;
            }
            _ => {}
        }
        i += 1;
    }

    match profile {
        BenchProfile::Fast => BenchConfig {
            profile,
            timesteps: 5,
            n_train_samples: 120,
            n_test_samples: 40,
            epochs: 8,
            eval_every: 2,
            use_feature_cache,
        },
        BenchProfile::Balanced => BenchConfig {
            profile,
            timesteps: 40,
            n_train_samples: 600,
            n_test_samples: 120,
            epochs: 25,
            eval_every: 5,
            use_feature_cache,
        },
        BenchProfile::Full => BenchConfig {
            profile,
            timesteps: 100,
            n_train_samples: 1000,
            n_test_samples: 200,
            epochs: 50,
            eval_every: 5,
            use_feature_cache,
        },
    }
}

fn main() {
    let cfg = parse_profile();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Hybrid Spiking Neural Network + MLP MNIST Benchmark     ║");
    println!("║                    Target: >=90% Accuracy                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let snn_layers = vec![28 * 28, 256, 128]; // MNIST 28×28 → hidden layers
    let readout_hidden = 256;
    let n_classes = 10;
    let timesteps = cfg.timesteps;
    let n_train_samples = cfg.n_train_samples;
    let n_test_samples = cfg.n_test_samples;
    let epochs = cfg.epochs;
    let lr_mlp = 0.05;
    let lr_snn = 0.001;

    println!("Architecture:");
    println!("  SNN Layers:      {}", format_vec(&snn_layers));
    println!("  MLP Readout:     {} → {} → {}", snn_layers.last().unwrap(), readout_hidden, n_classes);
    println!("  Profile:         {:?}", cfg.profile);
    println!("  Timesteps:       {}", timesteps);
    println!("\nTraining Configuration:");
    println!("  Train Samples:   {}", n_train_samples);
    println!("  Test Samples:    {}", n_test_samples);
    println!("  Epochs:          {}", epochs);
    println!("  Eval Every:      {}", cfg.eval_every);
    println!("  LR (MLP):        {}", lr_mlp);
    println!("  LR (SNN):        {}", lr_snn);
    println!("  Feature Cache:   {}", if cfg.use_feature_cache { "ON" } else { "OFF" });
    println!();

    // =========================================================================
    // Generate Synthetic MNIST-like Data
    // =========================================================================
    println!("Generating {:} training + {}: test samples...", n_train_samples, n_test_samples);

    let (train_inputs, train_labels) = generate_mnist_data(n_train_samples, snn_layers[0]);
    let (test_inputs, test_labels) = generate_mnist_data(n_test_samples, snn_layers[0]);

    println!("  ✓ Generated {} training samples", n_train_samples);
    println!("  ✓ Generated {} test samples\n", n_test_samples);

    // =========================================================================
    // Initialize Classifier
    // =========================================================================
    println!("Initializing HybridSpikingClassifier...");
    let mut classifier =
        HybridSpikingClassifier::new(&snn_layers, readout_hidden, n_classes, timesteps, 42.0);
    println!("  Total SNN parameters:  {}", classifier.snn.param_count());
    println!("  Total MLP parameters:  {}", classifier.readout.param_count());
    println!("  Total parameters:      {}\n", classifier.param_count());

    // =========================================================================
    // Training Loop
    // =========================================================================
    println!("Starting training ({} epochs)...", epochs);
    println!("────────────────────────────────────────────────────────────────");

    if cfg.use_feature_cache {
        println!("Note: static feature cache is disabled in hybrid mode (SNN updates every step).");
    }

    let train_start = Instant::now();
    let mut epoch_stats = Vec::new();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut num_samples = 0usize;

        // Training pass
        for sample in 0..n_train_samples {
            let input = &train_inputs[sample * snn_layers[0]..(sample + 1) * snn_layers[0]];
            let target = train_labels[sample];

            let loss = classifier.train_step(input, target, lr_mlp, lr_snn);
            total_loss += loss;
            num_samples += 1;
        }

        let avg_loss = total_loss / num_samples as f32;
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();

        // Evaluation
        if epoch % cfg.eval_every == 0 || epoch == epochs - 1 {
            let train_acc =
                classifier.accuracy(&train_inputs, &train_labels, n_train_samples, snn_layers[0]);
            let test_acc =
                classifier.accuracy(&test_inputs, &test_labels, n_test_samples, snn_layers[0]);

            println!(
                " Epoch {:3} | Loss: {:6.4} | Train Acc: {:5.1}% | Test Acc: {:5.1}% | {:6.2}s",
                epoch + 1,
                avg_loss,
                train_acc * 100.0,
                test_acc * 100.0,
                epoch_elapsed
            );

            epoch_stats.push((epoch + 1, avg_loss, train_acc, test_acc, epoch_elapsed));
        } else {
            // Just print loss without accuracy calculation
            println!(
                " Epoch {:3} | Loss: {:6.4} | (skip eval)            {:6.2}s",
                epoch + 1, avg_loss, epoch_elapsed
            );
        }
    }

    let total_train_time = train_start.elapsed().as_secs_f64();
    println!("────────────────────────────────────────────────────────────────");
    println!("Training complete in {:.2}s\n", total_train_time);

    // =========================================================================
    // Final Evaluation
    // =========================================================================
    println!("Final Evaluation:");
    let final_train_acc =
        classifier.accuracy(&train_inputs, &train_labels, n_train_samples, snn_layers[0]);
    let final_test_acc =
        classifier.accuracy(&test_inputs, &test_labels, n_test_samples, snn_layers[0]);

    println!("  Train Accuracy: {:.1}%", final_train_acc * 100.0);
    println!("  Test Accuracy:  {:.1}%", final_test_acc * 100.0);
    println!();

    // =========================================================================
    // Verdict
    // =========================================================================
    println!("╔═══════════════════════════════════════════════════════════════╗");
    if final_test_acc >= 0.90 {
        println!("║                    ✓ SUCCESS: >=90% Achieved!                ║");
    } else if final_test_acc >= 0.80 {
        println!("║                  ⚠ GOOD: {:.1}% (close to 90%)                ║", final_test_acc * 100.0);
    } else if final_test_acc >= 0.50 {
        println!("║                 ~ OKAY: {:.1}% (learning works)               ║", final_test_acc * 100.0);
    } else {
        println!("║                  ✗ POOR: {:.1}% (needs debugging)             ║", final_test_acc * 100.0);
    }
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

// =========================================================================
// Synthetic MNIST Data Generation
// =========================================================================

/// Generate synthetic MNIST-like data
/// Each class has a distinct pattern in input space.
fn generate_mnist_data(n_samples: usize, input_dim: usize) -> (Vec<f32>, Vec<u8>) {
    let mut inputs = Vec::with_capacity(n_samples * input_dim);
    let mut labels = Vec::with_capacity(n_samples);

    for sample in 0..n_samples {
        let class = (sample % 10) as u8;

        // Generate class-correlated pattern
        let input: Vec<f32> = (0..input_dim)
            .map(|i| {
                // Base pattern: depends on class
                let base = ((i as u8 + class * 7).wrapping_mul(13) % 17) as f32 / 17.0;

                // Add some noise
                let noise = ((i as u32 * 19 + sample as u32 * 31) % 100) as f32 / 100.0;

                (base * 0.7 + noise * 0.3).clamp(0.0, 1.0)
            })
            .collect();

        inputs.extend(input);
        labels.push(class);
    }

    (inputs, labels)
}

fn format_vec(v: &[usize]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" → ")
}

