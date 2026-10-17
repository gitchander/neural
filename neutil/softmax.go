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

func Softmax(logits, probabilities []float64) int {
	// return softmax_v1(logits, probabilities)
	// return softmax_v2(logits, probabilities)
	// return softmax_v3(logits, probabilities)
	// return softmax_v4(logits, probabilities)
	return softmax_v5(logits, probabilities)
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

func softmax_v1(logits, probabilities []float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}

	// For BCE:
	// $ go build -gcflags="-d=ssa/check_bce/debug=1" softmax.go

	// Bounds check elimination (BCE) hint.
	// Compiler should report "Found IsInBounds" here, removing checks in loops.
	_ = probabilities[n-1] // "Found IsInBounds"

	maxLogit := slices.Max(logits)
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

func softmax_v2(logits, probabilities []float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}
	if len(probabilities) < n {
		panic(errorHaveMin("softmax_v2", "probabilities length", len(probabilities), n))

	}
	maxLogit := slices.Max(logits)
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

func softmax_v3(logits, probabilities []float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}
	if len(probabilities) != n {
		panic(errorHaveWant("softmax_v3", "probabilities length", len(probabilities), n))
	}
	maxLogit := slices.Max(logits)
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

// softmax_v4 calculates the probabilities from the given logits and writes them into the provided buffer.
// The function operates on the intersection of the lengths of both slices (matching the behavior of copy).
// It returns the number of elements actually processed.
func softmax_v4(logits, probabilities []float64) int {

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
	invSum := 1.0 / sum
	for i := range n {
		probabilities[i] *= invSum
	}
	return n
}

func softmax_v5(logits, probabilities []float64) int {
	n := len(logits)
	if len(probabilities) != n {
		panic(errorHaveWant("softmax_v5", "probabilities length", len(probabilities), n))
	}
	softmaxKernel(logits, probabilities, n)
	return n
}

//------------------------------------------------------------------------------

func softmaxKernel(logits, probabilities []float64, n int) {

	if n == 0 {
		return
	}

	// BCE hint: eliminates bounds checks in the loops below.
	_ = logits[n-1]
	_ = probabilities[n-1]

	maxLogit := slices.Max(logits[:n])
	var sum float64
	for i := range n {
		probabilities[i] = math.Exp(logits[i] - maxLogit)
		sum += probabilities[i]
	}
	invSum := 1.0 / sum
	for i := range n {
		probabilities[i] *= invSum
	}
}

//------------------------------------------------------------------------------
