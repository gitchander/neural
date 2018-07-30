package neural

import (
	"math"
)

// The activation function
func sigmoid(x, a float64) float64 {
	return 1 / (1 + math.Exp(-2*a*x))
}

func sigmoidPrime(x, a float64) float64 {
	return 2 * a * x * (1 - x)
}
