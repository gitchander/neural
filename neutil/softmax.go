package neutil

import (
	"math"
)

// https://deepnotes.io/softmax-crossentropy

func Softmax(xs []float64) []float64 {
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

func SoftmaxStable(xs []float64) []float64 {
	if len(xs) == 0 {
		return nil
	}
	var (
		maxValue = maxFloat64s(xs)
		ys       = make([]float64, len(xs))
	)
	var sum float64
	for i, x := range xs {
		ys[i] = math.Exp(x - maxValue)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}
