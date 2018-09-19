package neural

import "math"

// https://en.wikipedia.org/wiki/Activation_function

// The activation function
type ActivationFunc interface {
	Func(x float64) float64        // f(x)
	Derivative(fx float64) float64 // {\frac {\partial f(x)}{\partial x}}
}

type Step struct{}

var _ ActivationFunc = Step{}

func (Step) Func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func (Step) Derivative(fx float64) float64 {
	return 0
}

type Linear struct{}

var _ ActivationFunc = Linear{}

func (Linear) Func(x float64) float64 {
	return x
}

func (Linear) Derivative(fx float64) float64 {
	return 1
}

type ReLU struct{}

var _ ActivationFunc = ReLU{}

func (ReLU) Func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func (ReLU) Derivative(fx float64) float64 {
	if fx > 0 {
		return 1
	}
	return 0
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

type Logistic struct {
	Alpha float64
}

var _ ActivationFunc = Logistic{}

func (p Logistic) Func(x float64) float64 {
	// {f(x) = {\frac {1}{1 + e^{-\alpha x}}}}
	return 1 / (1 + math.Exp(-p.Alpha*x))
}

func (p Logistic) Derivative(fx float64) float64 {
	// {{\frac {\partial f(x)}{\partial x}} = \alpha f(x) (1 - f(x))}
	return p.Alpha * fx * (1 - fx)
}

type Tanh struct{} // range (-1, 1)

var _ ActivationFunc = Tanh{}

func (Tanh) Func(x float64) float64 {
	var a = math.Exp(2 * x)
	return (a - 1) / (a + 1)
}

func (Tanh) Derivative(fx float64) float64 {
	return (1 - fx*fx)
}
