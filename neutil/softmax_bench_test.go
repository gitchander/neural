package neutil

import (
	"math/rand"
	"testing"
)

// go test -bench=. -benchmem

// Data preparation (1000 classes — standard dimension for ImageNet models)
const vectorSize = 1000

// generateLogits populates a slice with random values between 0.0 and 10.0
func generateLogits(size int) []float64 {
	logits := make([]float64, size)
	for i := range logits {
		logits[i] = rand.Float64() * 10.0
	}
	return logits
}

// ============================================================================
// API WRAPPER BENCHMARKS
// ============================================================================

// BenchmarkSoftmax_Buffer evaluates Softmax performance with a pre-allocated external buffer (0 allocations)
func BenchmarkSoftmax_Buffer(b *testing.B) {
	logits := generateLogits(vectorSize)
	probs := make([]float64, vectorSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(logits, probs)
	}
}

// BenchmarkSoftmaxReplace evaluates the In-Place mode performance (0 allocations, high cache locality)
func BenchmarkSoftmaxReplace(b *testing.B) {
	logits := generateLogits(vectorSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Logits convert to probabilities on the first iteration; acceptable for pure execution profiling.
		SoftmaxInPlace(logits)
	}
}

// BenchmarkSoftmaxAlloc evaluates performance when allocating a fresh slice on every invocation
func BenchmarkSoftmaxAlloc(b *testing.B) {
	logits := generateLogits(vectorSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SoftmaxWithAlloc(logits)
	}
}

// ============================================================================
// EXPERIMENTAL REVISIONS COMPARISON (v1, v2, v3)
// ============================================================================

func BenchmarkSoftmax_V1_Buffer(b *testing.B) {
	logits := generateLogits(vectorSize)
	probs := make([]float64, vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v1(logits, probs)
	}
}

func BenchmarkSoftmax_V1_InPlace(b *testing.B) {
	logits := generateLogits(vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v1(logits, logits)
	}
}

func BenchmarkSoftmax_V2_Buffer(b *testing.B) {
	logits := generateLogits(vectorSize)
	probs := make([]float64, vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v2(logits, probs)
	}
}

func BenchmarkSoftmax_V2_InPlace(b *testing.B) {
	logits := generateLogits(vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v2(logits, logits)
	}
}

func BenchmarkSoftmax_V3_Buffer(b *testing.B) {
	logits := generateLogits(vectorSize)
	probs := make([]float64, vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v3(logits, probs)
	}
}

func BenchmarkSoftmax_V3_InPlace(b *testing.B) {
	logits := generateLogits(vectorSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax_v3(logits, logits)
	}
}
