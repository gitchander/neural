package neutil

import (
	"math"
	"testing"
)

// go test -v

// TestSoftmaxMath verifies the mathematical accuracy and core properties of the Softmax function.
func TestSoftmaxMath(t *testing.T) {
	tests := []struct {
		name   string
		logits []float64
		want   []float64 // Exact mathematical outputs
	}{
		{
			name:   "Standard Distribution",
			logits: []float64{2.0, 1.0, 0.1},
			want:   []float64{0.6590011388859679, 0.24243297070471392, 0.09856589040931818},
		},
		{
			name:   "Equal Logits",
			logits: []float64{5.5, 5.5, 5.5},
			want:   []float64{0.3333333333333333, 0.3333333333333333, 0.3333333333333333},
		},
		{
			name:   "Numerical Stability (Large Numbers)",
			logits: []float64{1000.0, 999.0, 998.0},
			want:   []float64{0.6652409557748218, 0.24472847105479764, 0.09003057317038046},
		},
		{
			name:   "Negative Logits",
			logits: []float64{-1.0, -2.0, -3.0},
			want:   []float64{0.6652409557748218, 0.24472847105479764, 0.09003057317038046},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 1. Test the base Softmax function with a pre-allocated buffer
			probs := make([]float64, len(tt.logits))
			n := Softmax(tt.logits, probs)

			if n != len(tt.logits) {
				t.Errorf("Softmax returned processed count %d, want %d", n, len(tt.logits))
			}

			verifyResults(t, probs, tt.want)

			// 2. Test the allocating version SoftmaxWithAlloc
			probsAlloc := SoftmaxWithAlloc(tt.logits)
			verifyResults(t, probsAlloc, tt.want)

			// 3. Test the In-Place version (copying logits first to preserve the original test case)
			logitsCopy := append([]float64(nil), tt.logits...)
			SoftmaxInPlace(logitsCopy)
			verifyResults(t, logitsCopy, tt.want)
		})
	}
}

// verifyResults is a helper function to validate the sum invariant and value precision.
func verifyResults(t *testing.T, got, want []float64) {
	t.Helper()

	// Verification 1: The sum of probabilities must be exactly 1.0 (within float64 precision tolerance)
	sum := 0.0
	for _, p := range got {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Errorf("Sum of probabilities = %f, want 1.0", sum)
	}

	// Verification 2: Element-wise comparison with the expected output
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("At index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

// TestSoftmaxEdgeCases verifies the functions' behavior under boundary conditions and mismatched buffer sizes.
func TestSoftmaxEdgeCases(t *testing.T) {
	// 1. Empty slices (must return without panicking)
	t.Run("Empty Slices", func(t *testing.T) {
		n := Softmax([]float64{}, []float64{})
		if n != 0 {
			t.Errorf("Expected 0 for empty slices, got %d", n)
		}

		// Should successfully return an empty slice without panicking
		res := SoftmaxWithAlloc([]float64{})
		if len(res) != 0 {
			t.Errorf("Expected empty result from SoftmaxWithAlloc, got len %d", len(res))
		}
	})

	// 2. Mismatched buffer size behavior (copy function parity)
	t.Run("Buffer Mismatches Behavior", func(t *testing.T) {
		logits := []float64{2.0, 1.0, 0.1} // len = 3

		// Provided buffer is smaller than the logits slice
		smallProbs := make([]float64, 2)
		n1 := Softmax(logits, smallProbs)
		if n1 != 2 {
			t.Errorf("Expected to process 2 elements, got %d", n1)
		}
		// The sum of the processed subset must equal 1.0
		sum1 := smallProbs[0] + smallProbs[1]
		if math.Abs(sum1-1.0) > 1e-12 {
			t.Errorf("Subset sum = %f, want 1.0", sum1)
		}

		// Provided buffer is larger than the logits slice
		largeProbs := make([]float64, 5)
		n2 := Softmax(logits, largeProbs)
		if n2 != 3 {
			t.Errorf("Expected to process 3 elements, got %d", n2)
		}
		// Extra buffer trailing elements must remain untouched (0.0)
		if largeProbs[3] != 0.0 || largeProbs[4] != 0.0 {
			t.Errorf("Extra buffer elements were overwritten: %v", largeProbs)
		}
	})
}
