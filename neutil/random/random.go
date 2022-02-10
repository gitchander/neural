package random

import (
	"math/rand"
)

func FloatByInterval(r *rand.Rand, min, max float64) float64 {
	return lerp(min, max, r.Float64())
}
