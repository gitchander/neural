package neural

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// https://en.wikipedia.org/wiki/Activation_function

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

//------------------------------------------------------------------------------
type Sigmoid struct{}

var _ ActivationFunc = Sigmoid{}

// Sigmoid implements the sigmoid function
// for use in activation functions.
func (Sigmoid) Func(x float64) float64 {

	// latex:
	// {f(x) = {\frac {1}{1 + e^{-x}}}}

	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (Sigmoid) Derivative(fx float64) float64 {

	// latex:
	// {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}

	return fx * (1 - fx)
}

//------------------------------------------------------------------------------
type Logistic struct {
	Alpha float64
}

var _ ActivationFunc = Logistic{}

func (p Logistic) Func(x float64) float64 {

	// latex:
	// {f(x) = {\frac {1}{1 + e^{-\alpha x}}}}

	return 1 / (1 + math.Exp(-p.Alpha*x))
}

func (p Logistic) Derivative(fx float64) float64 {

	// latex:
	// {{\frac {\partial f(x)}{\partial x}} = \alpha f(x) (1 - f(x))}

	return p.Alpha * fx * (1 - fx)
}

//------------------------------------------------------------------------------
// range: (-1, 1)
type Tanh struct{}

var _ ActivationFunc = Tanh{}

func (Tanh) Func(x float64) float64 {

	// latex:
	// {f(x)=\frac{e^{2 x}-1}{e^{2 x}+1}}

	a := math.Exp(2 * x)
	return (a - 1) / (a + 1)
}

func (Tanh) Derivative(fx float64) float64 {

	// latex:
	// {{\frac {\partial f(x)}{\partial x}} = 1-f(x)^2}

	return (1 - fx*fx)
}

//------------------------------------------------------------------------------
func parseActivationFunc(s string) (ActivationFunc, error) {

	//--------------------------------
	// "linear"
	// "step"
	// "logistic:alpha=1"
	//--------------------------------

	vs := strings.Split(s, ":")

	if len(vs) < 1 {
		return nil, errors.New("parseActivationFunc: there are no instances")
	}

	fields, err := parseFields(vs[1])
	if err != nil {
		return nil, err
	}

	var af ActivationFunc

	switch vs[0] {
	case "linear":
		af = Linear{}
	case "step":
		af = Step{}
	case "relu":
		af = ReLU{}
	case "sigmoid":
		af = Sigmoid{}
	case "logistic":
		{
			const key = "alpha"
			text, ok := fields[key]
			if !ok {
				return nil, fmt.Errorf("there is no field <%s>", key)
			}
			alpha, err := strconv.ParseFloat(text, 64)
			if err != nil {
				return nil, fmt.Errorf("field <%s> parse error", key)
			}
			af = Logistic{Alpha: alpha}
		}
	case "tanh":
		af = Tanh{}
	default:
		return nil, fmt.Errorf("unknown activation func <%s>", vs[0])
	}

	return af, nil
}

func parseFields(s string) (map[string]string, error) {

	// example of format: "one=1,two=2,three=3"
	// example of format: "half=0.5,quarter=0.25,pi=3.14"

	m := make(map[string]string)
	vs := strings.Split(s, ",")
	for _, v := range vs {
		ws := strings.Split(v, "=")
		if len(ws) != 2 {
			return nil, errors.New("invalid field format")
		}
		m[ws[0]] = ws[1]
	}
	return m, nil
}
