//! Hyperdimensional Computing (HDC) — ternary vector algebra.
//!
//! Operations on 10,000-dim ternary vectors {-1, 0, +1}:
//! - Bind (XOR-like): encodes associations (dog + chases = relation)
//! - Bundle (majority vote): combines concepts (dog + cat = animals)
//! - Permute (cyclic shift): encodes order/position
//!
//! All operations are O(D) where D=dimensionality. No multiplication.
//! This is how brains might encode concepts — distributed, high-dimensional, robust.

/// A hyperdimensional ternary vector.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HdVector {
    /// Ternary components {-1, 0, +1} stored as i8
    pub data: Vec<i8>,
    pub dim: usize,
}

impl HdVector {
    /// Create a random HD vector with given seed.
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = seed;
        let data: Vec<i8> = (0..dim).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            match (rng >> 62) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            }
        }).collect();
        Self { data, dim }
    }

    /// Zero vector.
    pub fn zero(dim: usize) -> Self {
        Self { data: vec![0i8; dim], dim }
    }

    /// From f32 slice (ternarize).
    pub fn from_f32(values: &[f32]) -> Self {
        let dim = values.len();
        let mean_abs: f32 = values.iter().map(|v| v.abs()).sum::<f32>() / dim as f32;
        let threshold = mean_abs * 0.7;
        let data: Vec<i8> = values.iter().map(|&v| {
            if v > threshold { 1i8 } else if v < -threshold { -1i8 } else { 0i8 }
        }).collect();
        Self { data, dim }
    }

    /// Bind: element-wise ternary multiplication (XOR analog).
    /// Encodes association: bind(A, B) is the "relationship" between A and B.
    /// bind(bind(A, B), B) ≈ A (quasi-inverse)
    pub fn bind(&self, other: &HdVector) -> HdVector {
        assert_eq!(self.dim, other.dim);
        let data: Vec<i8> = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| (a as i16 * b as i16).max(-1).min(1) as i8)
            .collect();
        HdVector { data, dim: self.dim }
    }

    /// Bundle: element-wise sum + threshold (majority vote).
    /// Combines multiple concepts: bundle([dog, cat, fish]) ≈ "animals".
    pub fn bundle(vectors: &[&HdVector]) -> HdVector {
        if vectors.is_empty() { return HdVector::zero(0); }
        let dim = vectors[0].dim;
        let mut sums = vec![0i32; dim];
        for v in vectors {
            for i in 0..dim {
                sums[i] += v.data[i] as i32;
            }
        }
        // Threshold: majority vote
        let data: Vec<i8> = sums.iter().map(|&s| {
            if s > 0 { 1i8 } else if s < 0 { -1i8 } else { 0i8 }
        }).collect();
        HdVector { data, dim }
    }

    /// Permute: cyclic shift (encodes position/order).
    /// permute(A, 1) = A shifted by 1 position.
    pub fn permute(&self, shift: usize) -> HdVector {
        let mut data = vec![0i8; self.dim];
        for i in 0..self.dim {
            data[(i + shift) % self.dim] = self.data[i];
        }
        HdVector { data, dim: self.dim }
    }

    /// Cosine similarity (using integer dot product).
    pub fn similarity(&self, other: &HdVector) -> f32 {
        assert_eq!(self.dim, other.dim);
        let dot: i64 = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| a as i64 * b as i64)
            .sum();
        let norm_a: f64 = self.data.iter().map(|&a| (a as i64 * a as i64) as f64).sum::<f64>().sqrt();
        let norm_b: f64 = other.data.iter().map(|&b| (b as i64 * b as i64) as f64).sum::<f64>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
        (dot as f64 / (norm_a * norm_b)) as f32
    }

    /// Check if all elements are ternary.
    pub fn is_ternary(&self) -> bool {
        self.data.iter().all(|&v| v == -1 || v == 0 || v == 1)
    }
}

/// HDC Memory: stores concept vectors, retrieves by similarity.
pub struct HdMemory {
    pub items: Vec<(String, HdVector)>,
    pub dim: usize,
}

impl HdMemory {
    pub fn new(dim: usize) -> Self {
        Self { items: Vec::new(), dim }
    }

    /// Store a named concept.
    pub fn store(&mut self, name: &str, vector: HdVector) {
        self.items.push((name.to_string(), vector));
    }

    /// Query: find most similar stored concept.
    pub fn query(&self, vector: &HdVector) -> Option<(&str, f32)> {
        self.items.iter()
            .map(|(name, stored)| (name.as_str(), vector.similarity(stored)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    }

    /// Encode a sentence: bundle of permuted word vectors.
    pub fn encode_sequence(&self, word_vectors: &[&HdVector]) -> HdVector {
        let permuted: Vec<HdVector> = word_vectors.iter().enumerate()
            .map(|(pos, wv)| wv.permute(pos))
            .collect();
        let refs: Vec<&HdVector> = permuted.iter().collect();
        HdVector::bundle(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hd_vector_basic() {
        let v = HdVector::random(10000, 42);
        assert_eq!(v.dim, 10000);
        assert!(v.is_ternary());
        // Self-similarity = 1.0
        let sim = v.similarity(&v);
        assert!((sim - 1.0).abs() < 0.01, "Self-similarity should be ~1.0, got {}", sim);
    }

    #[test]
    fn hd_random_vectors_orthogonal() {
        let a = HdVector::random(10000, 42);
        let b = HdVector::random(10000, 137);
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.1, "Random HD vectors should be near-orthogonal, got {}", sim);
    }

    #[test]
    fn hd_bind_quasi_inverse() {
        let a = HdVector::random(10000, 42);
        let b = HdVector::random(10000, 137);
        // bind(a, b) then bind again with b should recover something similar to a
        let ab = a.bind(&b);
        let recovered = ab.bind(&b);
        let sim = a.similarity(&recovered);
        assert!(sim > 0.5, "bind(bind(a,b), b) should be similar to a, got {}", sim);
    }

    #[test]
    fn hd_bundle_combines() {
        let a = HdVector::random(10000, 1);
        let b = HdVector::random(10000, 2);
        let c = HdVector::random(10000, 3);
        let combined = HdVector::bundle(&[&a, &b, &c]);
        // Combined should be somewhat similar to each component
        assert!(combined.similarity(&a) > 0.2);
        assert!(combined.similarity(&b) > 0.2);
        assert!(combined.similarity(&c) > 0.2);
        assert!(combined.is_ternary());
    }

    #[test]
    fn hd_permute_encodes_order() {
        let a = HdVector::random(10000, 42);
        let p1 = a.permute(1);
        let p2 = a.permute(2);
        // Permuted versions should be different from original
        assert!(a.similarity(&p1) < 0.5, "Permute should change the vector");
        assert!(p1.similarity(&p2) < 0.5, "Different permutations should differ");
    }

    #[test]
    fn hd_memory_store_query() {
        let mut mem = HdMemory::new(10000);
        let dog = HdVector::random(10000, 1);
        let cat = HdVector::random(10000, 2);
        let car = HdVector::random(10000, 3);

        mem.store("dog", dog.clone());
        mem.store("cat", cat.clone());
        mem.store("car", car.clone());

        // Query with dog should return dog
        let (name, sim) = mem.query(&dog).unwrap();
        assert_eq!(name, "dog");
        assert!(sim > 0.9);

        // Query with something similar to dog+cat should return one of them
        let animal = HdVector::bundle(&[&dog, &cat]);
        let (name, _) = mem.query(&animal).unwrap();
        assert!(name == "dog" || name == "cat", "Should match an animal, got {}", name);
    }

    #[test]
    fn hd_sequence_encoding() {
        let mem = HdMemory::new(10000);
        let the = HdVector::random(10000, 10);
        let dog = HdVector::random(10000, 20);
        let runs = HdVector::random(10000, 30);

        // "the dog runs" vs "runs dog the" should be different
        let seq1 = mem.encode_sequence(&[&the, &dog, &runs]);
        let seq2 = mem.encode_sequence(&[&runs, &dog, &the]);
        let sim = seq1.similarity(&seq2);
        assert!(sim < 0.5, "Different word order should produce different encodings, got {}", sim);
    }
}
