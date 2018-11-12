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
		outputs:      make([]float64, len(p.outputLayer.ns)),
		costFunc:     costMeanSquared{},
	}
}

func (bp *BP) SetLearningRate(learningRate float64) {
	bp.learningRate = crop(learningRate)
}

func (bp *BP) LearnSample(sample Sample) {

	p := bp.p
	p.SetInputs(sample.Inputs)
	p.Calculate()

	//	if err = p.checkOutputs(sample.Outputs); err != nil {
	//		return err
	//	}
	var (
		lastIndex = len(p.layers) - 1
		lastLayer = p.layers[lastIndex]
	)
	for j, n := range lastLayer.ns {
		n.delta = p.actFunc.Derivative(n.out) * bp.costFunc.Derivative(sample.Outputs[j], n.out)
	}

	for k := lastIndex - 1; k > 0; k-- {
		var (
			layer     = p.layers[k]
			nextLayer = p.layers[k+1]
		)
		for j, n := range layer.ns {
			var sum float64
			for _, n_next := range nextLayer.ns {
				sum += n_next.delta * n_next.weights[j]
			}
			n.delta = p.actFunc.Derivative(n.out) * sum
		}
	}

	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer = p.layers[k-1]
			layer     = p.layers[k]
		)
		for _, n := range layer.ns {
			for i, n_prev := range prevLayer.ns {
				n.weights[i] -= bp.learningRate * n.delta * n_prev.out
			}
			n.bias -= bp.learningRate * n.delta * 1
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
		inLen  = len(p.inputLayer.ns)
		outLen = len(p.outputLayer.ns)
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
