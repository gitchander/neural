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

	last := len(p.layers) - 1

	var (
		x     = p.ssx[last+1]
		layer = p.layers[last]
	)
	for j, n := range layer.ns {
		n.delta = sigmoidPrime(x[j], p.a) * (x[j] - sample.Outputs[j])
	}

	for li := last - 1; li >= 0; li-- {
		var (
			x = p.ssx[li+1]

			layer     = p.layers[li]
			layerNext = p.layers[li+1]
		)

		for j, n := range layer.ns {
			var sum float64
			for _, n_next := range layerNext.ns {
				sum += n_next.delta * n_next.weights[j]
			}
			n.delta = sigmoidPrime(x[j], p.a) * sum
		}
	}

	for li, layer := range p.layers {

		var x = p.ssx[li]

		for _, n := range layer.ns {
			for i := range n.weights {
				n.weights[i] -= bp.learningRate * n.delta * x[i]
			}
			n.bias -= bp.learningRate * n.delta * 1
		}
	}

	return nil
}
