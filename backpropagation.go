package neural

import (
	"fmt"
)

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

// Backpropagation
type BP struct {
	p            *MLP
	ldeltas      [][]float64
	learningRate float64 // (0 <= learningRate <= 1)
	outputs      []float64
	costFunc     CostFunc
}

func NewBP(p *MLP, cf CostFunc) *BP {

	if cf == nil {
		cf = costMeanSquared{}
	}

	ldeltas := make([][]float64, len(p.layers))
	for i, layer := range p.layers {
		ldeltas[i] = make([]float64, len(layer.neurons))
	}

	return &BP{
		p:            p,
		ldeltas:      ldeltas,
		learningRate: 1,
		outputs:      make([]float64, len(p.getOutputLayer().neurons)),
		costFunc:     cf,
	}
}

func (bp *BP) SetLearningRate(learningRate float64) {
	bp.learningRate = clampFloat64(learningRate, 0, 1)
}

func (bp *BP) LearnSample(sample Sample) {

	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()

	var (
		lastIndex  = len(p.layers) - 1
		lastLayer  = p.layers[lastIndex]
		lastDeltas = bp.ldeltas[lastIndex]
	)
	for j, n := range lastLayer.neurons {
		var (
			afD = lastLayer.actFunc.Derivative(n.out)
			cfD = bp.costFunc.Derivative(sample.Outputs[j], n.out)
		)
		lastDeltas[j] = afD * cfD
	}

	for k := lastIndex - 1; k > 0; k-- {
		var (
			currLayer  = p.layers[k]
			currDeltas = bp.ldeltas[k]

			nextLayer  = p.layers[k+1]
			nextDeltas = bp.ldeltas[k+1]
		)
		for j, currNeuron := range currLayer.neurons {
			var sum float64
			for i, nextNeuron := range nextLayer.neurons {
				sum += nextDeltas[i] * nextNeuron.weights[j]
			}
			afD := currLayer.actFunc.Derivative(currNeuron.out)
			currDeltas[j] = afD * sum
		}
	}

	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer = p.layers[k-1]

			currLayer  = p.layers[k]
			currDeltas = bp.ldeltas[k]
		)
		for j, currNeuron := range currLayer.neurons {
			for i, prevNeuron := range prevLayer.neurons {
				currNeuron.weights[i] -= bp.learningRate * currDeltas[j] * prevNeuron.out
			}
			currNeuron.bias -= bp.learningRate * currDeltas[j] * 1
		}
	}
}

func (bp *BP) SampleCost(sample Sample) (cost float64) {
	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()
	p.GetOutputs(bp.outputs)
	return bp.costFunc.Func(sample.Outputs, bp.outputs)
}

func (bp *BP) LearnSamples(samples []Sample) (averageCost float64) {
	var sum float64
	for _, sample := range samples {
		bp.LearnSample(sample)
		sum += bp.SampleCost(sample)
	}
	averageCost = sum / float64(len(samples))
	return averageCost
}

func Learn(p *MLP, samples []Sample, learnRate float64, epochMax int,
	f func(epoch int, averageCost float64) bool) error {

	err := checkSamplesTopology(p, samples)
	if err != nil {
		return err
	}

	bp := NewBP(p, CFMeanSquared)
	bp.SetLearningRate(learnRate)
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost := bp.LearnSamples(samples)
		if !f(epoch, averageCost) {
			break
		}
	}

	return nil
}

func checkSamplesTopology(p *MLP, samples []Sample) error {

	format := "invalid sample (%d): wrong %s length (%d), must be (%d)"

	var (
		inLen  = len(p.getInputLayer().neurons)
		outLen = len(p.getOutputLayer().neurons)
	)

	for i, sample := range samples {
		if len(sample.Inputs) != inLen {
			return fmt.Errorf(format, i, "inputs", len(sample.Inputs), inLen)
		}
		if len(sample.Outputs) != outLen {
			return fmt.Errorf(format, i, "outputs", len(sample.Outputs), outLen)
		}
	}

	return nil
}
