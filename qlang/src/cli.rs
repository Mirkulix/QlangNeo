//! QLANG CLI — Train, Infer, Inspect ternary neural networks.
//!
//! Like BitNet.cpp, but with integrated training and agent communication.
//!
//! Usage:
//!   qlang train  --data mnist --epochs 15 --output model.qlbg
//!   qlang infer  --model model.qlbg --input "path/to/image"
//!   qlang info   model.qlbg
//!   qlang bench  model.qlbg

use std::time::Instant;

use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType};
use qlang_runtime::forward_forward::FFNetwork;
use qlang_runtime::mnist::MnistData;
use qlang_runtime::ternary_ops;

// ============================================================
// Model file format: .qlbg with embedded ternary weights
//
// Header:  QLBG graph (architecture description)
// Payload: packed 2-bit ternary weights + f32 biases + metadata
// ============================================================

/// Saved ternary model (serializable).
#[derive(serde::Serialize, serde::Deserialize)]
struct TernaryModel {
    /// Architecture: list of (in_dim, out_dim) per layer
    layers: Vec<(usize, usize)>,
    /// Packed 2-bit ternary weights per layer
    packed_weights: Vec<Vec<u8>>,
    /// Scaling factors per layer
    alphas: Vec<f32>,
    /// Biases per layer (f32)
    biases: Vec<Vec<f32>>,
    /// Number of classes
    n_classes: usize,
    /// Training metadata
    metadata: ModelMetadata,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ModelMetadata {
    method: String,
    epochs: usize,
    train_accuracy_f32: f32,
    train_accuracy_ternary: f32,
    train_samples: usize,
    test_samples: usize,
    train_time_secs: f64,
    total_params: usize,
    f32_size_bytes: usize,
    ternary_size_bytes: usize,
}

impl TernaryModel {
    /// Save to file (QLBG header + bincode payload).
    #[allow(dead_code)] // Save API retained for programmatic use
    fn save(&self, path: &str) -> Result<(), String> {
        // Build a QLANG graph describing the architecture
        let mut graph = Graph::new(format!("ternary_model_{}", self.n_classes));
        let str_type = TensorType::new(Dtype::Utf8, Shape::scalar());

        let input = graph.add_node(
            Op::Input { name: "image".into() },
            vec![], vec![str_type.clone()],
        );
        let output = graph.add_node(
            Op::Output { name: "class".into() },
            vec![str_type.clone()], vec![],
        );
        graph.add_edge(input, 0, output, 0, str_type);

        // Add metadata
        graph.metadata.insert("method".into(), self.metadata.method.clone());
        graph.metadata.insert("epochs".into(), self.metadata.epochs.to_string());
        graph.metadata.insert("accuracy_f32".into(), format!("{:.1}", self.metadata.train_accuracy_f32 * 100.0));
        graph.metadata.insert("accuracy_ternary".into(), format!("{:.1}", self.metadata.train_accuracy_ternary * 100.0));
        graph.metadata.insert("params".into(), self.metadata.total_params.to_string());
        graph.metadata.insert("compression".into(), format!("{:.1}x",
            self.metadata.f32_size_bytes as f64 / self.metadata.ternary_size_bytes as f64));

        // Write: QLBG header + payload
        let qlbg = binary::to_binary(&graph);
        let payload = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;

        let mut file_data = Vec::with_capacity(qlbg.len() + 4 + payload.len());
        file_data.extend_from_slice(&qlbg);
        file_data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&payload);

        std::fs::write(path, &file_data).map_err(|e| format!("write: {e}"))?;
        Ok(())
    }

    /// Load from file.
    /// Format: [QLBG header bytes][4-byte payload len][bincode payload]
    fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;

        // Find the payload: scan for the bincode marker after the QLBG header.
        // QLBG ends with a 32-byte SHA256 hash. We look for the payload length
        // field after skipping the QLBG magic + content.
        // Strategy: try parsing from different offsets until bincode succeeds.
        // The QLBG header size varies, but the payload length is right after it.

        // Quick approach: search backwards for valid bincode payload
        for offset in 50..data.len().min(2000) {
            if offset + 4 > data.len() { continue; }
            let payload_len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            let payload_start = offset + 4;
            if payload_start + payload_len == data.len() {
                // This looks right — payload fills exactly to end of file
                match bincode::deserialize::<TernaryModel>(&data[payload_start..]) {
                    Ok(model) => return Ok(model),
                    Err(_) => continue,
                }
            }
        }

        Err("could not find valid model payload in file".into())
    }

    /// Reconstruct FFNetwork from saved model.
    fn to_network(&self) -> FFNetwork {
        let layer_sizes: Vec<usize> = std::iter::once(self.layers[0].0)
            .chain(self.layers.iter().map(|l| l.1))
            .collect();
        let mut net = FFNetwork::new(&layer_sizes, self.n_classes);

        for (i, layer) in net.layers.iter_mut().enumerate() {
            let weights = ternary_ops::unpack_ternary(
                &self.packed_weights[i],
                self.layers[i].0 * self.layers[i].1,
                self.alphas[i],
            );
            layer.weights = weights;
            layer.biases = self.biases[i].clone();
        }

        net
    }

    /// Create from a trained FFNetwork.
    #[allow(dead_code)] // Constructor retained for programmatic use
    fn from_network(net: &FFNetwork, meta: ModelMetadata) -> Self {
        let mut layers = Vec::new();
        let mut packed_weights = Vec::new();
        let mut alphas = Vec::new();
        let mut biases = Vec::new();

        for layer in &net.layers {
            layers.push((layer.in_dim, layer.out_dim));
            let (packed, alpha) = ternary_ops::pack_ternary(&layer.weights);
            packed_weights.push(packed);
            alphas.push(alpha);
            biases.push(layer.biases.clone());
        }

        Self {
            layers,
            packed_weights,
            alphas,
            biases,
            n_classes: net.n_classes,
            metadata: meta,
        }
    }
}

// ============================================================
// CLI Commands
// ============================================================

fn cmd_train(data_path: &str, epochs: usize, output: &str, _hidden_layers: &[usize]) {
    use qlang_runtime::ternary_brain::TernaryBrain;

    println!("QLANG Train — Ternary Brain (Statistical Init + Competitive Hebbian)\n");

    // Load data
    let data = match MnistData::load_from_dir(data_path) {
        Ok(d) => {
            println!("Loaded REAL MNIST from '{}'", data_path);
            d
        }
        Err(_) => {
            println!("MNIST not found at '{}', using synthetic", data_path);
            MnistData::synthetic(5000, 1000)
        }
    };

    let train_limit = data.n_train.min(60000);
    let test_limit = data.n_test.min(10000);
    let train_images = &data.train_images[..train_limit * 784];
    let train_labels = &data.train_labels[..train_limit];
    let test_images = &data.test_images[..test_limit * 784];
    let test_labels = &data.test_labels[..test_limit];

    println!("Train: {}, Test: {}", train_limit, test_limit);

    // Phase 1: Statistical Initialization (instant)
    println!("\n=== Phase 1: Statistical Init ===");
    let total_start = Instant::now();
    let mut brain = TernaryBrain::init(train_images, train_labels, 784, train_limit, 10, 20);
    let phase1_time = total_start.elapsed();
    let phase1_acc = brain.accuracy(test_images, test_labels, test_limit);
    println!("  Accuracy: {:.1}% (in {:?})", phase1_acc * 100.0, phase1_time);

    // Phase 2: Competitive Hebbian Refinement
    println!("\n=== Phase 2: Competitive Hebbian ({} rounds) ===", epochs);
    for round in 0..epochs {
        brain.refine(train_images, train_labels, train_limit, 784);
        if round % 3 == 0 || round == epochs - 1 {
            let acc = brain.accuracy(test_images, test_labels, test_limit);
            println!("  Round {:>2}/{}: {:.1}% ({:.1?})",
                round + 1, epochs, acc * 100.0, total_start.elapsed());
        }
    }

    let train_time = total_start.elapsed();
    let final_acc = brain.accuracy(test_images, test_labels, test_limit);
    let total_weights = brain.total_weights();
    let ternary_verified = brain.verify_ternary();

    // Ternary size: 2 bits per weight
    let tern_size = total_weights * 2 / 8;
    let f32_size = total_weights * 4;

    println!("\n=== RESULT ===");
    println!("  Method:    Ternary Brain (Statistical Init + Competitive Hebbian)");
    println!("  Accuracy:  {:.1}%", final_acc * 100.0);
    println!("  Weights:   {} (all ternary: {})", total_weights, ternary_verified);
    println!("  f32 size:  {:.1} KB", f32_size as f64 / 1024.0);
    println!("  tern size: {:.1} KB", tern_size as f64 / 1024.0);
    println!("  compress:  {:.1}x", f32_size as f64 / tern_size as f64);
    println!("  time:      {:.1}s", train_time.as_secs_f64());
    println!("  gradients: ZERO");

    // Save model
    let meta = ModelMetadata {
        method: "ternary-brain".into(),
        epochs,
        train_accuracy_f32: final_acc, // brain is 100% ternary, no f32
        train_accuracy_ternary: final_acc,
        train_samples: train_limit,
        test_samples: test_limit,
        train_time_secs: train_time.as_secs_f64(),
        total_params: total_weights,
        f32_size_bytes: f32_size,
        ternary_size_bytes: tern_size,
    };

    // Simple save: metadata JSON + raw ternary weights
    let meta_json = serde_json::to_vec(&serde_json::json!({
        "method": meta.method,
        "accuracy": final_acc,
        "total_weights": total_weights,
        "epochs": epochs,
        "train_time_secs": train_time.as_secs_f64(),
        "compression": f32_size as f64 / tern_size as f64,
    })).unwrap_or_default();

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&[0x51, 0x4C, 0x54, 0x42]); // "QLTB" = QLANG Ternary Brain
    file_data.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    file_data.extend_from_slice(&meta_json);

    // Save TernaryBrain weights using the brain's built-in serialization
    match brain.save(output) {
        Ok(()) => {
            println!("\nSaved TernaryBrain: {} ({} bytes)", output, tern_size);
        }
        Err(e) => {
            eprintln!("Failed to save brain weights: {e}, falling back to metadata-only");
            // Fallback: just save metadata
            match std::fs::write(output, &file_data) {
                Ok(()) => println!("\nSaved (metadata only): {}", output),
                Err(e) => eprintln!("Save failed: {e}"),
            }
        }
    }
}

fn cmd_infer(model_path: &str, input_text: &str) {
    println!("QLANG Infer — Ternary Zero-Multiply\n");

    // First try loading as TernaryBrain (bincode format from cmd_train)
    if let Ok(brain) = qlang_runtime::ternary_brain::TernaryBrain::load(model_path) {
        let neurons = brain.total_weights() / brain.image_dim.max(1);
        println!("Loaded TernaryBrain: {} neurons, {} classes",
            neurons, brain.n_classes);

        let data = MnistData::synthetic(100, 100);

        // Single sample inference
        if let Ok(digit) = input_text.parse::<u8>() {
            if digit < 10 {
                println!("\nTesting with synthetic digit {}...", digit);
                for i in 0..data.n_test {
                    if data.test_labels[i] == digit {
                        let image = &data.test_images[i * 784..(i + 1) * 784];
                        let start = Instant::now();
                        let preds = brain.predict(image, 1);
                        let infer_time = start.elapsed();
                        println!("Predicted: {} (expected: {}) in {:?}", preds[0], digit, infer_time);
                        return;
                    }
                }
            }
        }

        // Batch inference
        println!("\nRunning batch inference on synthetic test data...");
        let start = Instant::now();
        let acc = brain.accuracy(&data.test_images, &data.test_labels, data.n_test.min(500));
        let infer_time = start.elapsed();
        println!("Accuracy: {:.1}% on {} samples", acc * 100.0, data.n_test.min(500));
        println!("Time: {:?}", infer_time);
        return;
    }

    // Fall back to TernaryModel (FFNetwork-based)
    let model = match TernaryModel::load(model_path) {
        Ok(m) => m,
        Err(e) => { eprintln!("Load failed: {e}"); return; }
    };

    println!("Model: {} layers, {} classes", model.layers.len(), model.n_classes);
    println!("Method: {}", model.metadata.method);
    println!("Accuracy: {:.1}% (ternary)", model.metadata.train_accuracy_ternary * 100.0);

    let net = model.to_network();

    // If input is a digit (0-9), generate a test sample
    if let Ok(digit) = input_text.parse::<u8>() {
        if digit < 10 {
            println!("\nGenerating synthetic test image for digit {}...", digit);
            let data = MnistData::synthetic(100, 100);
            // Find a sample with this label
            for i in 0..data.n_test {
                if data.test_labels[i] == digit {
                    let image = &data.test_images[i * 784..(i + 1) * 784];
                    let start = Instant::now();
                    let preds = net.predict_ternary(image, 784, 1);
                    let infer_time = start.elapsed();
                    println!("Predicted: {} (expected: {}) in {:?}", preds[0], digit, infer_time);
                    let correct = preds[0] == digit;
                    println!("Result: {}", if correct { "CORRECT" } else { "WRONG" });
                    return;
                }
            }
        }
    }

    // Batch inference on MNIST test set
    println!("\nRunning batch inference on synthetic test data...");
    let data = MnistData::synthetic(100, 500);
    let start = Instant::now();
    let acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
    let infer_time = start.elapsed();

    let (f32_ops, tern_ops, zeros) = ternary_ops::ops_saved(&net.layers[0].weights);
    println!("Accuracy: {:.1}% on {} samples", acc * 100.0, data.n_test);
    println!("Time: {:?} ({:.1}us per sample)", infer_time,
        infer_time.as_micros() as f64 / data.n_test as f64);
    println!("Ops saved: {:.0}% (zeros: {:.0}%)",
        (1.0 - tern_ops as f64 / f32_ops as f64) * 100.0,
        zeros as f64 / net.layers[0].weights.len() as f64 * 100.0);
}

fn cmd_info(model_path: &str) {
    println!("QLANG Info\n");

    let data = match std::fs::read(model_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("Read failed: {e}"); return; }
    };

    let file_size = data.len();
    println!("File: {} ({} bytes)", model_path, file_size);

    // Parse model payload directly (skips QLBG hash issue)
    match TernaryModel::load(model_path) {
        Ok(model) => {
            println!("\nModel:");
            println!("  Method:    {}", model.metadata.method);
            println!("  Classes:   {}", model.n_classes);
            println!("  Layers:    {}", model.layers.len());
            for (i, (in_d, out_d)) in model.layers.iter().enumerate() {
                let n_weights = in_d * out_d;
                let packed_bytes = model.packed_weights[i].len();
                println!("    [{}] {}x{} = {} weights ({} bytes packed, alpha={:.3})",
                    i, in_d, out_d, n_weights, packed_bytes, model.alphas[i]);
            }
            println!("\nTraining:");
            println!("  Epochs:    {}", model.metadata.epochs);
            println!("  Samples:   {} train / {} test", model.metadata.train_samples, model.metadata.test_samples);
            println!("  Time:      {:.1}s", model.metadata.train_time_secs);
            println!("  f32 acc:   {:.1}%", model.metadata.train_accuracy_f32 * 100.0);
            println!("  tern acc:  {:.1}%", model.metadata.train_accuracy_ternary * 100.0);
            println!("\nSize:");
            println!("  f32:     {:.1} KB", model.metadata.f32_size_bytes as f64 / 1024.0);
            println!("  ternary: {:.1} KB", model.metadata.ternary_size_bytes as f64 / 1024.0);
            println!("  ratio:   {:.1}x", model.metadata.f32_size_bytes as f64 / model.metadata.ternary_size_bytes as f64);
        }
        Err(e) => eprintln!("Payload parse failed: {e}"),
    }
}

fn cmd_bench(model_path: &str) {
    println!("QLANG Bench — Ternary vs f32 Inference\n");

    let model = match TernaryModel::load(model_path) {
        Ok(m) => m,
        Err(e) => { eprintln!("Load failed: {e}"); return; }
    };

    let net = model.to_network();
    let data = MnistData::synthetic(100, 1000);

    // Warm up
    let _ = net.accuracy_ternary(&data.test_images[..784], &data.test_labels[..1], 784, 1);

    // Ternary inference
    let start = Instant::now();
    let tern_acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
    let tern_time = start.elapsed();

    // f32 inference
    let start = Instant::now();
    let f32_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
    let f32_time = start.elapsed();

    println!("Samples: {}", data.n_test);
    println!();
    println!("{:<20} {:>10} {:>12} {:>10}", "Method", "Accuracy", "Total", "Per sample");
    println!("{}", "-".repeat(54));
    println!("{:<20} {:>9.1}% {:>12.1?} {:>9.1}us", "Ternary (add/sub)",
        tern_acc * 100.0, tern_time, tern_time.as_micros() as f64 / data.n_test as f64);
    println!("{:<20} {:>9.1}% {:>12.1?} {:>9.1}us", "f32 (shadow)",
        f32_acc * 100.0, f32_time, f32_time.as_micros() as f64 / data.n_test as f64);
    println!();

    let speedup = f32_time.as_nanos() as f64 / tern_time.as_nanos() as f64;
    println!("Ternary speedup: {:.2}x", speedup);

    let total_weights: usize = net.layers.iter().map(|l| l.weights.len()).sum();
    let (_, _, zeros) = ternary_ops::ops_saved(&net.layers[0].weights);
    let zero_pct = zeros as f64 / net.layers[0].weights.len() as f64 * 100.0;
    println!("Total weights: {} ({:.0}% zeros)", total_weights, zero_pct);
    println!("File size: {} bytes", std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0));
}

fn cmd_tci_demo(rounds: usize) {
    use qlang_runtime::tci::TciEngine;
    use qlang_runtime::ternary_brain::TernaryBrain;

    println!("QLANG TCI Demo — Ternary Continual Intelligence\n");

    let image_dim = 784usize;
    let data = MnistData::synthetic(2500, 400);

    let template = TernaryBrain::init(
        &data.train_images,
        &data.train_labels,
        image_dim,
        1200,
        10,
        12,
    );
    let mut engine = TciEngine::new(template);

    let n_a = 1000usize;
    let acc_a = engine
        .learn_task(
            "digits-normal",
            &data.train_images[..n_a * image_dim],
            &data.train_labels[..n_a],
            n_a,
            rounds,
        )
        .unwrap_or(0.0);

    let n_b = 1000usize;
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
            rounds,
        )
        .unwrap_or(0.0);

    let test_n = data.n_test.min(300);
    let consensus_acc = engine
        .evaluate_consensus(&data.test_images[..test_n * image_dim], &data.test_labels[..test_n], test_n)
        .unwrap_or(0.0);
    let task0_acc = engine
        .evaluate_task(0, &data.test_images[..test_n * image_dim], &data.test_labels[..test_n], test_n)
        .unwrap_or(0.0);
    let task1_acc = engine
        .evaluate_task(1, &data.test_images[..test_n * image_dim], &data.test_labels[..test_n], test_n)
        .unwrap_or(0.0);

    let sample_normal = &data.test_images[..image_dim];
    let routed_normal = engine.route_and_predict(sample_normal).ok();

    let mut sample_inverted = sample_normal.to_vec();
    for v in &mut sample_inverted {
        *v = 1.0 - *v;
    }
    let routed_inverted = engine.route_and_predict(&sample_inverted).ok();

    println!("Learned tasks: {}", engine.task_count());
    println!("Task names: {}", engine.task_names().join(", "));
    println!("Task train acc: normal={:.1}% inverted={:.1}%", acc_a * 100.0, acc_b * 100.0);
    println!(
        "Retention (test): task0={:.1}% task1={:.1}% consensus={:.1}%",
        task0_acc * 100.0,
        task1_acc * 100.0,
        consensus_acc * 100.0
    );

    if let Some(r) = routed_normal {
        println!(
            "Route normal sample -> task='{}' sim={:.3} pred={} conf={:.3}",
            r.task_name, r.similarity, r.prediction, r.confidence
        );
    }
    if let Some(r) = routed_inverted {
        println!(
            "Route inverted sample -> task='{}' sim={:.3} pred={} conf={:.3}",
            r.task_name, r.similarity, r.prediction, r.confidence
        );
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => {
            let data = arg_value(&args, "--data").unwrap_or_else(|| "data/mnist".into());
            let epochs: usize = arg_value(&args, "--epochs").and_then(|s| s.parse().ok()).unwrap_or(15);
            let output = arg_value(&args, "--output").unwrap_or_else(|| "model.qlbg".into());
            let hidden_str = arg_value(&args, "--hidden").unwrap_or_else(|| "512,256,128,64".into());
            let hidden: Vec<usize> = hidden_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            cmd_train(&data, epochs, &output, &hidden);
        }
        "infer" => {
            let model = arg_value(&args, "--model").unwrap_or_else(|| "model.qlbg".into());
            let input = arg_value(&args, "--input").unwrap_or_else(|| "5".into());
            cmd_infer(&model, &input);
        }
        "info" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("model.qlbg");
            cmd_info(path);
        }
        "bench" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("model.qlbg");
            cmd_bench(path);
        }
        "tci" => {
            let rounds: usize = arg_value(&args, "--rounds").and_then(|s| s.parse().ok()).unwrap_or(5);
            cmd_tci_demo(rounds);
        }
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("QLANG — Ternary Neural Network Engine");
    println!();
    println!("Like BitNet.cpp, but with integrated training and agent communication.");
    println!("Train, compress, and run inference with zero-multiply ternary weights.");
    println!();
    println!("USAGE:");
    println!("  qlang train  --data <mnist_dir> --epochs <n> --output <model.qlbg>");
    println!("  qlang infer  --model <model.qlbg> --input <digit|path>");
    println!("  qlang info   <model.qlbg>");
    println!("  qlang bench  <model.qlbg>");
    println!("  qlang tci    --rounds <n>");
    println!();
    println!("EXAMPLES:");
    println!("  qlang train --data data/mnist --epochs 15 --output digit.qlbg");
    println!("  qlang infer --model digit.qlbg --input 7");
    println!("  qlang info digit.qlbg");
    println!("  qlang bench digit.qlbg");
    println!("  qlang tci --rounds 5");
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
}
