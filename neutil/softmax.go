package neutil

import (
	"math"
	"slices"
)

//------------------------------------------------------------------------------

// https://deepnotes.io/softmax-crossentropy

// Inputs - Logits
// Outputs - Probabilities, normalized probabilities, Outputs, softmax scores

// probs - probabilities

//------------------------------------------------------------------------------

// The Softmax Function:

// Softmax calculates the probabilities from the given logits and writes them into the provided buffer.
// The function operates on the intersection of the lengths of both slices (matching the behavior of copy).
// It returns the number of elements actually processed.
func Softmax(logits, probabilities []float64) int {

	n := min(len(logits), len(probabilities))
	if n == 0 {
		return 0
	}

	// Slicing logits to n ensures extra elements don't affect the maximum logit calculation.
	// This also signals the Go compiler to eliminate boundary checks inside the loops (BCE optimization).
	maxLogit := slices.Max(logits[:n])

	var sum float64
	for i := range n {
		probabilities[i] = math.Exp(logits[i] - maxLogit)
		sum += probabilities[i]
	}
	for i := range n {
		probabilities[i] /= sum
	}
	return n
}

// SoftmaxInPlace calculates the probabilities directly inside the provided slice,
// overwriting the logits with the normalized probabilities. This mode maximizes CPU cache locality.

func SoftmaxInPlace(logits []float64) {
	_ = Softmax(logits, logits)
}

// SoftmaxWithAlloc creates a new slice of the exact required size, calculates the
// probabilities into it, and returns it. The original logits slice remains untouched.
func SoftmaxWithAlloc(logits []float64) []float64 {
	probabilities := make([]float64, len(logits))
	_ = Softmax(logits, probabilities)
	return probabilities
}

//------------------------------------------------------------------------------

// Internal experimental versions kept for historical comparison and reference.

func softmax_v1(logits, probs []float64) {
	n := len(logits)
	if n == 0 {
		return
	}
	maxLogit := slices.Max(logits)
	var sum float64
	for i := range n {
		probs[i] = math.Exp(logits[i] - maxLogit)
		sum += probs[i]
	}
	for i := range n {
		probs[i] /= sum
	}
}

func softmax_v2(logits, probs []float64) {
	n := len(logits)
	if n == 0 {
		return
	}
	if len(probs) != n {
		panic("softmax: length mismatch between logits and probabilities")
	}
	maxLogit := slices.Max(logits)
	var sum float64
	for i := range n {
		probs[i] = math.Exp(logits[i] - maxLogit)
		sum += probs[i]
	}
	for i := range n {
		probs[i] /= sum
	}
}

func softmax_v3(logits, probabilities []float64) int {
	n := min(len(logits), len(probabilities))
	if n == 0 {
		return 0
	}
	maxLogit := slices.Max(logits[:n])
	var sum float64
	for i := range n {
		probabilities[i] = math.Exp(logits[i] - maxLogit)
		sum += probabilities[i]
	}
	for i := range n {
		probabilities[i] /= sum
	}
	return n
}
