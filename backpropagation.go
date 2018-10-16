package neural

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
	bp.learningRate = crop_01(learningRate)
}

func (bp *BP) LearnSample(sample Sample) error {

	p := bp.p
	err := p.SetInputs(sample.Inputs)
	if err != nil {
		return err
	}
	p.Calculate()

	if err = p.checkOutputs(sample.Outputs); err != nil {
		return err
	}
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

	return nil
}

func (bp *BP) SampleCost(sample Sample) (cost float64) {

	p := bp.p
	err := p.SetInputs(sample.Inputs)
	if err != nil {
		panic(err)
	}

	p.Calculate()

	err = p.GetOutputs(bp.outputs)
	if err != nil {
		panic(err)
	}

	return bp.costFunc.Func(sample.Outputs, bp.outputs)
}

func (bp *BP) LearnSamples(samples []Sample) (averageCost float64, err error) {
	var sum float64
	for _, sample := range samples {
		err = bp.LearnSample(sample)
		if err != nil {
			return 0, err
		}
		sum += bp.SampleCost(sample)
	}
	averageCost = sum / float64(len(samples))
	return averageCost, nil
}

func Learn(p *MLP, samples []Sample, learnRate float64, epochMax int,
	f func(epoch int, averageCost float64) bool) error {

	bp := NewBP(p)
	bp.SetLearningRate(learnRate)
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost, err := bp.LearnSamples(samples)
		if err != nil {
			return err
		}
		if !f(epoch, averageCost) {
			break
		}
	}
	return nil
}
