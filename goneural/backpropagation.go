package goneural

import (
	"github.com/gitchander/neural/neutil"
)

// https://en.wikipedia.org/wiki/Backpropagation

// Backpropagation
type BP struct {
	p            *Neural
	ldeltas      [][]float64
	learningRate float64 // (0 <= learningRate <= 1)
	cc           *costCore
}

func NewBP(p *Neural, costFunc CostFunc) *BP {

	isCrossEntropy := p.getOutputLayer().act.isSoftmax
	cc := newCostCore(isCrossEntropy, costFunc)

	ldeltas := make([][]float64, len(p.layers))
	for i, layer := range p.layers {
		ldeltas[i] = make([]float64, len(layer.neurons))
	}

	return &BP{
		p:            p,
		ldeltas:      ldeltas,
		learningRate: 1,
		cc:           cc,
	}
}

func (bp *BP) SetLearningRate(learningRate float64) {
	bp.learningRate = clampFloat64(learningRate, 0, 1)
}

func (bp *BP) calcDeltas(sample Sample) {
	var (
		p = bp.p

		lastIndex  = len(p.layers) - 1
		lastLayer  = p.layers[lastIndex]
		lastDeltas = bp.ldeltas[lastIndex]
	)
	bp.cc.calcLastDeltas(sample, lastLayer, lastDeltas)

	for k := lastIndex - 1; k > 0; k-- {
		var (
			currLayer  = p.layers[k]
			currDeltas = bp.ldeltas[k]

			nextLayer  = p.layers[k+1]
			nextDeltas = bp.ldeltas[k+1]
		)
		for j := range currLayer.neurons {
			var sum float64
			for i, nextNeuron := range nextLayer.neurons {
				sum += nextDeltas[i] * nextNeuron.weights[j]
			}
			afD := currLayer.act.af.Derivative(currLayer.outputs[j])
			currDeltas[j] = afD * sum
		}
	}
}

func (bp *BP) updateWeights() {
	p := bp.p
	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer  = p.layers[k-1]
			currLayer  = p.layers[k]
			currDeltas = bp.ldeltas[k]
		)
		currLayer.updateWeights(prevLayer.outputs, currDeltas, bp.learningRate)
	}
}

func (bp *BP) TrainSample(sample Sample) float64 {

	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()

	outputs := p.getOutputLayer().outputs
	sampleCost := bp.cc.calcSampleCost(sample.Outputs, outputs)

	bp.calcDeltas(sample)
	bp.updateWeights()

	return sampleCost
}

func (bp *BP) TrainSamples(samples []Sample) (averageCost float64) {
	var sum float64
	for _, sample := range samples {
		sum += bp.TrainSample(sample)
	}
	averageCost = sum / float64(len(samples))
	return averageCost
}

//------------------------------------------------------------------------------

func (bp *BP) TestSample(sample Sample) (sampleCost float64, isCorrect bool) {

	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()

	outputs := p.getOutputLayer().outputs
	sampleCost = bp.cc.calcSampleCost(sample.Outputs, outputs)

	var (
		indexIdeal = neutil.IndexOfMax(sample.Outputs)
		indexReal  = neutil.IndexOfMax(outputs)
	)

	isCorrect = indexIdeal == indexReal

	return sampleCost, isCorrect
}

type TestResult struct {
	AverageCost float64
	ErrorRate   float64
	Accuracy    float64
}

func (bp *BP) TestSamples(samples []Sample) TestResult {
	var (
		sum        float64
		wrongCount int
	)
	for _, sample := range samples {
		sampleCost, isCorrect := bp.TestSample(sample)
		if !isCorrect {
			wrongCount++
		}
		sum += sampleCost
	}
	var (
		averageCost = sum / float64(len(samples))
		errorRate   = 100.0 * float64(wrongCount) / float64(len(samples))
	)
	return TestResult{
		AverageCost: averageCost,
		ErrorRate:   errorRate,
		Accuracy:    100.0 - errorRate,
	}
}
