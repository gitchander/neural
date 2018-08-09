package neural

import (
	"math/rand"
	"time"
)

func NewRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
}

func randRange(r *rand.Rand, a, b float64) float64 {
	return a + (b-a)*r.Float64()
}
