package neural

import (
	"math"
)

// https://deepnotes.io/softmax-crossentropy

func softmax(xs []float64) []float64 {
	ys := make([]float64, len(xs))
	var sum float64
	for i, x := range xs {
		ys[i] = math.Exp(x)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}

func softmaxStable(xs []float64) []float64 {
	index := IndexOfMax(xs)
	if index == -1 {
		return nil
	}
	max := xs[index]
	ys := make([]float64, len(xs))
	var sum float64
	for i, x := range xs {
		ys[i] = math.Exp(x - max)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}
