package goneural

import (
	"fmt"
	"math"

	"github.com/gitchander/neural/neutil"
	"github.com/gitchander/neural/neutil/baserw"
)

// https://en.wikipedia.org/wiki/Activation_function

// ActivationSpec, ActivationConfig
type ActivationConfig struct {
	AF        ActivationFunc // Is nil if IsSoftmax is true
	IsSoftmax bool
}

func (ac *ActivationConfig) Activate(logits []float64) {

	// If current activation function is Softmax
	if ac.IsSoftmax {
		neutil.SoftmaxInPlace(logits)
	} else {
		for i := range logits {
			logits[i] = ac.AF.Func(logits[i])
		}
	}
}

//------------------------------------------------------------------------------

// type activationContext struct {
// 	af        ActivationFunc
// 	isSoftmax bool // ActivationFunc is Softmax
// }

// func makeActivationContext(af ActivationConfig) activationContext {
// 	return activationContext{
// 		af: af,
// 	}
// }

// func (ac *activationContext) Activate(logits []float64) {

// 	// If current activation function is Softmax
// 	if ac.isSoftmax {
// 		neutil.SoftmaxInPlace(logits)
// 	} else {
// 		for i := range logits {
// 			logits[i] = ac.af.Func(logits[i])
// 		}
// 	}
// }

//------------------------------------------------------------------------------

func readActivationConfig(br *baserw.BaseReader) (*ActivationConfig, error) {

	name, err := br.ReadString()
	if err != nil {
		return nil, fmt.Errorf("read ActivationFunc name error: %w", err)
	}

	var ac ActivationConfig
	switch name {
	case "linear":
		ac = ActivationConfig{
			AF: Linear,
		}
	case "step":
		ac = ActivationConfig{
			AF: Step,
		}
	case "relu":
		ac = ActivationConfig{
			AF: ReLU,
		}
	case "logistic":
		{
			alpha, err := br.ReadFloat64()
			if err != nil {
				return nil, fmt.Errorf("read logistic.alpha error: %w", err)
			}
			ac = ActivationConfig{
				AF: actLogistic{alpha: alpha},
			}
		}
	case "sigmoid":
		ac = ActivationConfig{
			AF: actSigmoid{},
		}
	case "tanh":
		ac = ActivationConfig{
			AF: actTanh{},
		}
	case "softmax":
		ac = ActivationConfig{
			IsSoftmax: true,
		}
	default:
		return nil, fmt.Errorf("Invalid activation function name %q", name)
	}

	return &ac, nil
}

func writeActivationConfig(bw *baserw.BaseWriter, ac ActivationConfig) error {

	if ac.IsSoftmax {
		return bw.WriteString("softmax")
	}

	switch t := ac.AF.(type) {
	case actLinear:
		return bw.WriteString("linear")
	case actStep:
		return bw.WriteString("step")
	case actReLU:
		return bw.WriteString("relu")
	case actSigmoid:
		return bw.WriteString("sigmoid")
	case actTanh:
		return bw.WriteString("tanh")
	case actLogistic:
		{
			err := bw.WriteString("logistic")
			if err != nil {
				return err
			}
			return bw.WriteFloat64(t.alpha)
		}
	default:
		return fmt.Errorf("invalid activation name (%T)", t)
	}
}

//------------------------------------------------------------------------------

// The activation function
type ActivationFunc interface {
	Func(x float64) float64        // f(x)
	Derivative(fx float64) float64 // {\frac {\partial f(x)}{\partial x}}
}

//------------------------------------------------------------------------------

type actLinear struct{}

var _ ActivationFunc = actLinear{}

func (actLinear) Func(x float64) float64 {
	return x
}

func (actLinear) Derivative(fx float64) float64 {
	return 1
}

// Linear or Identity
var Linear ActivationFunc = actLinear{}

//------------------------------------------------------------------------------

type actStep struct{}

var _ ActivationFunc = actStep{}

func (actStep) Func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func (actStep) Derivative(fx float64) float64 {
	return 0
}

// Binary Step
var Step ActivationFunc = actStep{}

//------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

type actReLU struct{}

var _ ActivationFunc = actReLU{}

// max(0, x)
func (actReLU) Func(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (actReLU) Derivative(fx float64) float64 {
	if fx > 0 {
		return 1
	}
	return 0
}

var ReLU ActivationFunc = actReLU{}

//------------------------------------------------------------------------------

type actLogistic struct {
	alpha float64
}

var _ ActivationFunc = actLogistic{}

func (p actLogistic) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-\alpha x}}}}

	return 1 / (1 + math.Exp(-p.alpha*x))
}

func (p actLogistic) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = \alpha f(x) (1 - f(x))}

	return p.alpha * fx * (1 - fx)
}

func Logistic(alpha float64) ActivationFunc {
	if alpha <= 0 {
		alpha = 1.0
	}
	return actLogistic{alpha: alpha}
}

//------------------------------------------------------------------------------

type actSigmoid struct{}

var _ ActivationFunc = actSigmoid{}

// Sigmoid implements the sigmoid function
// for use in activation functions.
func (actSigmoid) Func(x float64) float64 {

	// latex: {f(x) = {\frac {1}{1 + e^{-x}}}}

	return 1 / (1 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func (actSigmoid) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = f(x)(1 - f(x))}

	return fx * (1 - fx)
}

var Sigmoid ActivationFunc = actSigmoid{}

//------------------------------------------------------------------------------

// range: (-1, 1)
type actTanh struct{}

var _ ActivationFunc = actTanh{}

func (actTanh) Func(x float64) float64 {

	// latex: {f(x)=\frac{e^{2 x}-1}{e^{2 x}+1}}

	a := math.Exp(2 * x)
	return (a - 1) / (a + 1)
}

func (actTanh) Derivative(fx float64) float64 {

	// latex: {{\frac {\partial f(x)}{\partial x}} = 1-f(x)^2}

	return (1 - fx*fx)
}

var Tanh ActivationFunc = actTanh{}

//------------------------------------------------------------------------------
