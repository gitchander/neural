package other

import (
	"math"
	"math/rand"
)

// (-1 < w < +1)
func randWeight(r *rand.Rand) float64 {
	return 2*r.Float64() - 1
}

type ActivationFunc interface {
	F(x float64) float64
	Df(fx float64) float64
}

type Sigmoid struct{}

var _ ActivationFunc = Sigmoid{}

// sigmoid implements the sigmoid function
// for use in activation functions.
func (Sigmoid) F(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (Sigmoid) Df(fx float64) float64 {
	return fx * (1 - fx)
}

type Logistic struct {
	Alpha float64
}

var _ ActivationFunc = Logistic{}

func (p Logistic) F(x float64) float64 {
	return 1 / (1 + math.Exp(-p.Alpha*x))
}

func (p Logistic) Df(fx float64) float64 {
	return p.Alpha * fx * (1 - fx)
}
