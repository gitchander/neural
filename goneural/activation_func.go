package goneural

import (
	"fmt"
	"math"
)

// https://en.wikipedia.org/wiki/Activation_function

// https://www.parasdahal.com/softmax-crossentropy

type actFuncExt struct {
	af        ActivationFunc
	isSoftmax bool // ActivationFunc is Softmax
}

func makeActivationFunc(ac ActivationConfig) (*actFuncExt, error) {

	var afe actFuncExt

	switch ac.Name {
	case "linear":
		afe.af = Linear{}
	case "step":
		afe.af = Step{}
	case "relu":
		afe.af = ReLU{}
	case "logistic":
		if ps := ac.Params; len(ps) > 0 {
			afe.af = Logistic{
				Alpha: ps[0],
			}
		} else {
			// Alpha = 1
			afe.af = Sigmoid{}
		}
	case "sigmoid":
		afe.af = Sigmoid{}
	case "tanh":
		afe.af = Tanh{}
	case "softmax":
		afe.af = Linear{}
		afe.isSoftmax = true
	default:
		return nil, fmt.Errorf("Invalid activation function %q", ac.Name)
	}

	return &afe, nil
}

//------------------------------------------------------------------------------

// The activation function
type ActivationFunc interface {
	Func(x float64) float64        // f(x)
	Derivative(fx float64) float64 // {\frac {\partial f(x)}{\partial x}}
}

//------------------------------------------------------------------------------

// Linear or Identity
type Linear struct{}

var _ ActivationFunc = Linear{}

func (Linear) Func(x float64) float64 {
	return x
}

func (Linear) Derivative(fx float64) float64 {
	return 1
}

//------------------------------------------------------------------------------

// Binary Step
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

//------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

type ReLU struct{}

var _ ActivationFunc = ReLU{}

// max(0, x)
func (ReLU) Func(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (ReLU) Derivative(fx float64) float64 {
	if fx > 0 {
		return 1
	}
	return 0
}

//------------------------------------------------------------------------------

type Logistic struct {
	Alpha float64
}

var _ ActivationFunc = Logistic{}

func (p Logistic) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-\alpha x}}}}

	return 1 / (1 + math.Exp(-p.Alpha*x))
}

func (p Logistic) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = \alpha f(x) (1 - f(x))}

	return p.Alpha * fx * (1 - fx)
}

//------------------------------------------------------------------------------

type Sigmoid struct{}

var _ ActivationFunc = Sigmoid{}

// Sigmoid implements the sigmoid function
// for use in activation functions.
func (Sigmoid) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-x}}}}

	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (Sigmoid) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}

	return fx * (1 - fx)
}

//------------------------------------------------------------------------------

// range: (-1, 1)
type Tanh struct{}

var _ ActivationFunc = Tanh{}

func (Tanh) Func(x float64) float64 {

	// latex: {f(x)=\frac{e^{2 x}-1}{e^{2 x}+1}}

	a := math.Exp(2 * x)
	return (a - 1) / (a + 1)
}

func (Tanh) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = 1-f(x)^2}

	return (1 - fx*fx)
}

//------------------------------------------------------------------------------
