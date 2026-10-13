package goneural

import (
	"math"
	"slices"
)

// The Softmax Function:

func softmax(logits, ys []float64) {
	if len(logits) == 0 {
		return
	}
	if len(ys) < len(logits) {
		panic("softmax: ys buffer is too small")
	}
	maxLogit := slices.Max(logits)
	var sum float64
	for i := range logits {
		ys[i] = math.Exp(logits[i] - maxLogit)
		sum += ys[i]
	}
	for i := range logits {
		ys[i] /= sum
	}
}

func softmaxAlloc(logits []float64) []float64 {
	ys := make([]float64, len(logits))
	softmax(logits, ys)
	return ys
}
