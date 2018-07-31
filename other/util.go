package other

import (
	"math"
	"math/rand"
)

// (-1 < w < +1)
func randWeight(r *rand.Rand) float64 {
	return 2*r.Float64() - 1
}

// The activation function
func sigmoid(x, a float64) float64 {
	return 1 / (1 + math.Exp(-2*a*x))
}

func sigmoidPrime(x, a float64) float64 {
	return 2 * a * x * (1 - x)
}
