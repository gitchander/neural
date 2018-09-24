package neural

import (
	"math/rand"
)

func randWeight(r *rand.Rand) float64 {
	return randRange(r, -0.5, 0.5)
}

func randRange(r *rand.Rand, min, max float64) float64 {
	return min + (max-min)*r.Float64()
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

// https://en.wikipedia.org/wiki/One-hot
// OneHot(3, 0): [1, 0, 0]
// OneHot(3, 1): [0, 1, 0]
// OneHot(3, 2): [0, 0, 1]
// OneHot(4, 0): [1, 0, 0, 0]
// OneHot(4, 3): [0, 0, 0, 1]
func OneHot(n, i int) []float64 {
	vs := make([]float64, n)
	vs[i] = 1
	return vs
}

func NormalizeInputs(samples []Sample) {

	var mins, maxs []float64

	for k, sample := range samples {
		var vs = sample.Inputs
		if k == 0 {
			n := len(vs)
			mins = make([]float64, n)
			maxs = make([]float64, n)
			for i, v := range vs {
				mins[i] = v
				maxs[i] = v
			}
		} else {
			for i, v := range vs {
				if v < mins[i] {
					mins[i] = v
				}
				if v > maxs[i] {
					maxs[i] = v
				}
			}
		}
	}

	for _, sample := range samples {
		var vs = sample.Inputs
		for i := range vs {
			vs[i] = normalize(vs[i], mins[i], maxs[i])
		}
	}
}

func normalize(x, min, max float64) float64 {
	return (x - min) / (max - min)
}

func IndexOfMin(vs []float64) (min int) {
	n := len(vs)
	if n == 0 {
		return -1
	}
	for i := 1; i < n; i++ {
		if vs[i] < vs[min] {
			min = i
		}
	}
	return min
}

func IndexOfMax(vs []float64) (max int) {
	n := len(vs)
	if n == 0 {
		return -1
	}
	for i := 1; i < n; i++ {
		if vs[max] < vs[i] {
			max = i
		}
	}
	return max
}
