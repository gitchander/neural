package neural

import (
	"math/rand"

	"github.com/gitchander/minmax"
)

type intRange struct {
	Min, Max float64
}

func randWeight(r *rand.Rand) float64 {
	return randRange(r, intRange{Min: -0.5, Max: 0.5})
}

func randRange(r *rand.Rand, e intRange) float64 {
	return lerp(e.Min, e.Max, r.Float64())
}

func lerp(v0, v1 float64, t float64) float64 {
	return v0*(1-t) + v1*t
}

func cropFloat64(x float64, min, max float64) float64 {
	if x < min {
		x = min
	}
	if x > max {
		x = max
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

	n := len(samples)
	if n == 0 {
		return
	}

	var vs = samples[0].Inputs
	var rs = make([]intRange, len(vs))
	for i, v := range vs {
		rs[i] = intRange{Min: v, Max: v}
	}

	for k := 1; k < n; k++ {
		var vs = samples[k].Inputs
		for i, v := range vs {
			if v < rs[i].Min {
				rs[i].Min = v
			}
			if v > rs[i].Max {
				rs[i].Max = v
			}
		}
	}

	for _, sample := range samples {
		var vs = sample.Inputs
		for i, v := range vs {
			vs[i] = normalize(v, rs[i])
		}
	}
}

func normalize(x float64, r intRange) float64 {
	return (x - r.Min) / (r.Max - r.Min)
}

type float64Slice []float64

func (v float64Slice) Len() int           { return len(v) }
func (v float64Slice) Less(i, j int) bool { return v[i] < v[j] }

func IndexOfMin(vs []float64) (min int) {
	return minmax.IndexOfMin(float64Slice(vs))
}

func IndexOfMax(vs []float64) (max int) {
	return minmax.IndexOfMax(float64Slice(vs))
}
