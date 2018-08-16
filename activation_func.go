package neural

import (
	"math"
)

// https://en.wikipedia.org/wiki/Activation_function

// The activation function
type activationFunc interface {
	Func(x float64) float64 // f(x)

	// {\frac {\partial f(x)}{\partial x}}
	Derivative(fx float64) float64
}

type sigmoid struct{}

var _ activationFunc = sigmoid{}

// sigmoid implements the sigmoid function
// for use in activation functions.
func (sigmoid) Func(x float64) float64 {
	// {f(x) = {\frac {1}{1 + e^{-x}}}}
	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (sigmoid) Derivative(fx float64) float64 {
	// {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}
	return fx * (1 - fx)
}

type sigmoidAlpha struct {
	Alpha float64
}

var _ activationFunc = sigmoidAlpha{}

func (p sigmoidAlpha) Func(x float64) float64 {
	// {f(x) = {\frac {1}{1 + e^{-2 \alpha x}}}}
	return 1 / (1 + math.Exp(-2*p.Alpha*x))
}

func (p sigmoidAlpha) Derivative(fx float64) float64 {
	// {{\frac {\partial f(x)}{\partial x}} = 2 \alpha f(x) (1 - f(x))}
	return 2 * p.Alpha * fx * (1 - fx)
}

type tanH struct{} // range (-1, 1)

var _ activationFunc = tanH{}

func (p tanH) Func(x float64) float64 {
	var (
		ex  = math.Exp(x)
		e_x = math.Exp(-x)
	)
	return (ex - e_x) / (ex + e_x)
}

func (p tanH) Derivative(fx float64) float64 {
	return (1 - fx*fx)
}
