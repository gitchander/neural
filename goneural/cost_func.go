package goneural

import (
	"github.com/gitchander/neural/neutil"
)

// loss function
// cost function

type CostFunc interface {

	// t - ideal outputs
	// x - real outputs
	Func(t, x []float64) float64

	// t[i] - ideal output [0 or 1]
	// x[i] - real output [0..1]
	Derivative(ti, xi float64) float64
}

//------------------------------------------------------------------------------

type costMeanSquared struct{}

func (costMeanSquared) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {
		delta := t[i] - x[i]
		sum += delta * delta
	}
	return sum / 2
}

func (costMeanSquared) Derivative(ti, xi float64) float64 {
	return xi - ti
}

var MeanSquared CostFunc = costMeanSquared{}

//------------------------------------------------------------------------------

type costCore struct {
	isCrossEntropy bool
	costFunc       CostFunc
}

func newCostCore(isCrossEntropy bool, costFunc CostFunc) *costCore {
	if !(isCrossEntropy) && (costFunc == nil) {
		costFunc = MeanSquared
	}
	return &costCore{
		isCrossEntropy: isCrossEntropy,
		costFunc:       costFunc,
	}
}

func (p *costCore) calcSampleCost(t, x []float64) float64 {
	var sampleCost float64
	if p.isCrossEntropy {
		sampleCost, _ = neutil.CrossEntropy(t, x)
	} else {
		sampleCost = p.costFunc.Func(t, x)
	}
	return sampleCost
}

func (p *costCore) calcLastDeltas(sample Sample, lastLayer *layer, lastDeltas []float64) {
	for j := range lastLayer.outputs {
		if p.isCrossEntropy {
			// Derivative of Cross Entropy Loss with Softmax
			lastDeltas[j] = neutil.CrossEntropySoftmaxDerivative(sample.Outputs[j], lastLayer.outputs[j])
		} else {
			var (
				out = lastLayer.outputs[j]
				afD = lastLayer.act.af.Derivative(out)
				cfD = p.costFunc.Derivative(sample.Outputs[j], out)
			)
			lastDeltas[j] = afD * cfD
		}
	}
}
