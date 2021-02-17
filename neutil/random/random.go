package random

import (
	"math/rand"
	"time"
)

func NewRandSeed(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

func NewRandTime(t time.Time) *rand.Rand {
	return NewRandSeed(t.UnixNano())
}

func NewRandNow() *rand.Rand {
	return NewRandTime(time.Now())
}

func FloatByInterval(r *rand.Rand, min, max float64) float64 {
	return lerp(min, max, r.Float64())
}

func lerp(v0, v1 float64, t float64) float64 {
	return v0*(1-t) + v1*t
}
