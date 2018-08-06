package neural

import (
	"math"
)

// https://en.wikipedia.org/wiki/Activation_function

// The activation function
type ActivationFunc interface {
	Func(x float64) float64 // f(x)

	// {\frac {\partial f(x)}{\partial x}}
	Derivative(fx float64) float64
}

type Sigmoid struct{}

var _ ActivationFunc = Sigmoid{}

// sigmoid implements the sigmoid function
// for use in activation functions.
func (Sigmoid) Func(x float64) float64 {
	// {f(x) = {\frac {1}{1 + e^{-x}}}}
	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (Sigmoid) Derivative(fx float64) float64 {
	// {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}
	return fx * (1 - fx)
}

type SigmoidAlpha struct {
	Alpha float64
}

var _ ActivationFunc = SigmoidAlpha{}

func (p SigmoidAlpha) Func(x float64) float64 {
	// {f(x) = {\frac {1}{1 + e^{-2 \alpha x}}}}
	return 1 / (1 + math.Exp(-2*p.Alpha*x))
}

func (p SigmoidAlpha) Derivative(fx float64) float64 {
	// {{\frac {\partial f(x)}{\partial x}} = 2 \alpha f(x) (1 - f(x))}
	return 2 * p.Alpha * fx * (1 - fx)
}

type TanH struct{}

// range (-1, 1)

var _ ActivationFunc = TanH{}

func (p TanH) Func(x float64) float64 {
	var (
		ex  = math.Exp(x)
		e_x = math.Exp(-x)
	)
	return (ex - e_x) / (ex + e_x)
}

func (p TanH) Derivative(fx float64) float64 {
	return (1 - fx*fx)
}
