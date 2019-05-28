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
	learningRate float64 // (0 <= learningRate <= 1)
	outputs      []float64
	costFunc     CostFunc
}

func NewBP(p *MLP) *BP {
	return &BP{
		p:            p,
		learningRate: 1,
		outputs:      make([]float64, len(p.outputLayer.neurons)),
		costFunc:     costMeanSquared{},
	}
}

func (bp *BP) SetLearningRate(learningRate float64) {
	bp.learningRate = cropFloat64(learningRate, 0, 1)
}

func (bp *BP) LearnSample(sample Sample) {

	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()

	var (
		lastIndex = len(p.layers) - 1
		lastLayer = p.layers[lastIndex]
	)
	for j, n := range lastLayer.neurons {
		n.delta = lastLayer.actFunc.Derivative(n.out) * bp.costFunc.Derivative(sample.Outputs[j], n.out)
	}

	for k := lastIndex - 1; k > 0; k-- {
		var (
			currLayer = p.layers[k]
			nextLayer = p.layers[k+1]
		)
		for j, currNeuron := range currLayer.neurons {
			var sum float64
			for _, nextNeuron := range nextLayer.neurons {
				sum += nextNeuron.delta * nextNeuron.weights[j]
			}
			currNeuron.delta = currLayer.actFunc.Derivative(currNeuron.out) * sum
		}
	}

	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer = p.layers[k-1]
			currLayer = p.layers[k]
		)
		for _, currNeuron := range currLayer.neurons {
			for i, prevNeuron := range prevLayer.neurons {
				currNeuron.weights[i] -= bp.learningRate * currNeuron.delta * prevNeuron.out
			}
			currNeuron.bias -= bp.learningRate * currNeuron.delta * 1
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

	bp := NewBP(p)
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
		inLen  = len(p.inputLayer.neurons)
		outLen = len(p.outputLayer.neurons)
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
