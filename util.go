package neural

import (
	"math/rand"
)

// mean squared error
// https://en.wikipedia.org/wiki/Mean_squared_error
func MSE(a, b []float64) float64 {
	n := len(a)
	if n != len(b) {
		panic("length not equal")
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return sum / float64(n)
}

func Mean(xs []float64) float64 {
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

// (-0.5 < w < +0.5)
func randWeight(r *rand.Rand) float64 {
	return r.Float64() - 0.5
}

func newMatrix2(n, m int) [][]float64 {
	ssv := make([][]float64, n)
	for i := range ssv {
		ssv[i] = make([]float64, m)
	}
	return ssv
}

func crop_01(x float64) float64 {
	if x < 0 {
		x = 0
	}
	if x > 1 {
		x = 1
	}
	return x
}
