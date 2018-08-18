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
