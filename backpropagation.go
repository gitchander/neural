package neural

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

type Backpropagation struct {
	p            *Perceptron
	learningRate float64 // (0 <= learningRate <= 1)
}

func NewBackpropagation(p *Perceptron) *Backpropagation {
	return &Backpropagation{
		p:            p,
		learningRate: 1,
	}
}

func (bp *Backpropagation) SetLearningRate(learningRate float64) {
	bp.learningRate = crop_01(learningRate)
}

func (bp *Backpropagation) Learn(sample Sample) error {

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
		n.delta = p.af.Derivative(n.out) * (n.out - sample.Outputs[j])
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
			n.delta = p.af.Derivative(n.out) * sum
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

func (bp *Backpropagation) LearnSamples(samples []Sample) (mse float64, err error) {
	var worst float64
	for _, sample := range samples {
		err := bp.Learn(sample)
		if err != nil {
			return 0, err
		}
		mse := bp.p.CalculateMSE(sample)
		if mse > worst {
			worst = mse
		}
	}
	return worst, nil
}
