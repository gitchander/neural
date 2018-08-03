package neural

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

type Backpropagation struct {
	learningRate float64 // (0 <= learningRate <= 1)
}

func NewBackpropagation() *Backpropagation {
	return &Backpropagation{
		learningRate: 1,
	}
}

func (bp *Backpropagation) SetLearningRate(learningRate float64) {
	bp.learningRate = crop_01(learningRate)
}

func (bp *Backpropagation) Learn(p *Perceptron, sample Sample) error {

	err := p.SetInputs(sample.Inputs)
	if err != nil {
		return err
	}
	p.Calculate()

	var (
		lastIndex = len(p.layers) - 1
		last      = p.layers[lastIndex]
	)
	for j, n := range last.ns {
		n.delta = sigmoidPrime(n.out, p.a) * (n.out - sample.Outputs[j])
	}

	for k := lastIndex - 1; k > 0; k-- {
		var (
			layer     = p.layers[k]
			layerNext = p.layers[k+1]
		)
		for j, n := range layer.ns {
			var sum float64
			for _, n_next := range layerNext.ns {
				sum += n_next.delta * n_next.weights[j]
			}
			n.delta = sigmoidPrime(n.out, p.a) * sum
		}
	}

	for k := 1; k < len(p.layers); k++ {
		var (
			layer     = p.layers[k]
			layerPrev = p.layers[k-1]
		)
		for _, n := range layer.ns {
			for i, n_prev := range layerPrev.ns {
				n.weights[i] -= bp.learningRate * n.delta * n_prev.out
			}
			n.bias -= bp.learningRate * n.delta * 1
		}
	}

	return nil
}
