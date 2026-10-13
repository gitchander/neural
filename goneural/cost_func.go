package goneural

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

type CrossEntropy struct{}

func (CrossEntropy) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {
		sum += t[i] * safeLog(x[i])
	}
	return -sum
}

// Derivative of Cross Entropy Loss with Softmax
func (CrossEntropy) SoftmaxDerivative(ti, xi float64) float64 {
	return xi - ti
}

//------------------------------------------------------------------------------

type BinaryCrossEntropy struct{}

func (BinaryCrossEntropy) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {

		// if (t[i] == 1) -> -log(x[i])
		// if (t[i] == 0) -> -log(1-x[i])

		sum -= (t[i] * safeLog(x[i])) + ((1 - t[i]) * safeLog(1-x[i]))
	}
	return sum
}

//------------------------------------------------------------------------------

type costCore struct {
	isCrossEntropy bool
	crossEntropy   CrossEntropy
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
		sampleCost = p.crossEntropy.Func(t, x)
	} else {
		sampleCost = p.costFunc.Func(t, x)
	}
	return sampleCost
}

func (p *costCore) calcLastDeltas(sample Sample, lastLayer *layer, lastDeltas []float64) {
	for j := range lastLayer.outputs {
		if p.isCrossEntropy {
			// Derivative of Cross Entropy Loss with Softmax
			lastDeltas[j] = p.crossEntropy.SoftmaxDerivative(sample.Outputs[j], lastLayer.outputs[j])
		} else {
			var (
				out = lastLayer.outputs[j]
				afD = lastLayer.afe.af.Derivative(out)
				cfD = p.costFunc.Derivative(sample.Outputs[j], out)
			)
			lastDeltas[j] = afD * cfD
		}
	}
}
