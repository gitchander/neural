package goneural

import (
	"fmt"
	"math"
)

// https://en.wikipedia.org/wiki/Activation_function

type MakerAF interface {
	MakeAF(params []float64) (ActivationFunc, error)
}

var afMap = newSyncMap()

func init() {
	afMap.Set("linear", makerAF_Linear{})
	afMap.Set("step", makerAF_Step{})
	afMap.Set("relu", makerAF_ReLU{})
	afMap.Set("logistic", makerAF_Logistic{})
	afMap.Set("sigmoid", makerAF_Sigmoid{})
	afMap.Set("tanh", makerAF_Tanh{})
	afMap.Set("softmax", makerAF_Softmax{})
}

func MakeActivationFunc(ac ActivationConfig) (ActivationFunc, error) {

	v, ok := afMap.Get(ac.Name)
	if !ok {
		return nil, fmt.Errorf("There is no activation func %q", ac.Name)
	}

	maf, ok := v.(MakerAF)
	if !ok {
		return nil, fmt.Errorf("bad convert type: (%T) -> (%T)", v, maf)
	}

	af, err := maf.MakeAF(ac.Params)
	if err != nil {
		return nil, err
	}

	return af, nil
}

//------------------------------------------------------------------------------

// The activation function
type ActivationFunc interface {
	Func(x float64) float64        // f(x)
	Derivative(fx float64) float64 // {\frac {\partial f(x)}{\partial x}}

	// IsSoftmax() bool
}

//------------------------------------------------------------------------------

// Linear or Identity
type af_Linear struct{}

var _ ActivationFunc = af_Linear{}

func (af_Linear) Func(x float64) float64 {
	return x
}

func (af_Linear) Derivative(fx float64) float64 {
	return 1
}

type makerAF_Linear struct{}

func (makerAF_Linear) MakeAF(params []float64) (ActivationFunc, error) {
	return af_Linear{}, nil
}

//------------------------------------------------------------------------------

// Binary Step
type af_Step struct{}

var _ ActivationFunc = af_Step{}

func (af_Step) Func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func (af_Step) Derivative(fx float64) float64 {
	return 0
}

type makerAF_Step struct{}

func (v makerAF_Step) MakeAF(params []float64) (ActivationFunc, error) {
	return af_Step{}, nil
}

//------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

type af_ReLU struct{}

var _ ActivationFunc = af_ReLU{}

// max(0, x)

func (af_ReLU) Func(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (af_ReLU) Derivative(fx float64) float64 {
	if fx > 0 {
		return 1
	}
	return 0
}

type makerAF_ReLU struct{}

func (v makerAF_ReLU) MakeAF(params []float64) (ActivationFunc, error) {
	return af_ReLU{}, nil
}

//------------------------------------------------------------------------------

type af_Logistic struct {
	Alpha float64
}

var _ ActivationFunc = af_Logistic{}

func (p af_Logistic) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-\alpha x}}}}

	return 1 / (1 + math.Exp(-p.Alpha*x))
}

func (p af_Logistic) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = \alpha f(x) (1 - f(x))}

	return p.Alpha * fx * (1 - fx)
}

type makerAF_Logistic struct{}

func (v makerAF_Logistic) MakeAF(params []float64) (ActivationFunc, error) {

	if len(params) < 1 {
		return nil, fmt.Errorf("There is no param")
	}
	alpha := params[0]

	af := af_Logistic{
		Alpha: alpha,
	}

	return af, nil
}

//------------------------------------------------------------------------------

type af_Sigmoid struct{}

var _ ActivationFunc = af_Sigmoid{}

// Sigmoid implements the sigmoid function
// for use in activation functions.
func (af_Sigmoid) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-x}}}}

	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (af_Sigmoid) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}

	return fx * (1 - fx)
}

type makerAF_Sigmoid struct{}

func (makerAF_Sigmoid) MakeAF(params []float64) (ActivationFunc, error) {
	return af_Sigmoid{}, nil
}

//------------------------------------------------------------------------------

// range: (-1, 1)
type af_Tanh struct{}

var _ ActivationFunc = af_Tanh{}

func (af_Tanh) Func(x float64) float64 {

	// latex: {f(x)=\frac{e^{2 x}-1}{e^{2 x}+1}}

	a := math.Exp(2 * x)
	return (a - 1) / (a + 1)
}

func (af_Tanh) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = 1-f(x)^2}

	return (1 - fx*fx)
}

type makerAF_Tanh struct{}

func (makerAF_Tanh) MakeAF(params []float64) (ActivationFunc, error) {
	return af_Tanh{}, nil
}

//------------------------------------------------------------------------------

// https://www.parasdahal.com/softmax-crossentropy

type af_Softmax struct{}

var _ ActivationFunc = af_Softmax{}

func (af_Softmax) Func(x float64) float64 {
	return math.Exp(x) // this is just nominator of softmax equation
}

func (af_Softmax) Derivative(fx float64) float64 {
	return fx * (1 - fx)
}

type makerAF_Softmax struct{}

func (makerAF_Softmax) MakeAF(params []float64) (ActivationFunc, error) {
	return af_Softmax{}, nil
}

//------------------------------------------------------------------------------
