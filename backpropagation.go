package neural

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

type Backpropagation struct {
	p            *Perceptron
	ssd          [][]float64 // delta Weight
	learningRate float64     // (0 <= learningRate <= 1)
}

func NewBackpropagation(p *Perceptron) *Backpropagation {
	ssd := make([][]float64, len(p.layers))
	for k, layer := range p.layers {
		ssd[k] = make([]float64, len(layer.ssw))
	}
	return &Backpropagation{
		p:            p,
		ssd:          ssd,
		learningRate: 1,
	}
}

func (bp *Backpropagation) SetLearningRate(learningRate float64) {
	bp.learningRate = crop_01(learningRate)
}

// outputs - ideal outputs
func (bp *Backpropagation) Learn(sample Sample) error {

	p := bp.p

	err := p.SetInputs(sample.Inputs)
	if err != nil {
		return err
	}
	p.Calculate()

	ssd := bp.ssd
	last := len(ssd) - 1

	var (
		x     = p.ssx[last+1]
		delta = ssd[last]
	)
	for j := range delta {
		delta[j] = sigmoidPrime(x[j], p.a) * (x[j] - sample.Outputs[j])
	}

	for li := last - 1; li >= 0; li-- {
		var (
			x     = p.ssx[li+1]
			delta = ssd[li]

			deltaNext   = ssd[li+1]
			weightsNext = p.layers[li+1].ssw
		)

		for j := range delta {
			var sum float64
			for k := range deltaNext {
				sum += deltaNext[k] * weightsNext[k][j]
			}
			delta[j] = sigmoidPrime(x[j], p.a) * sum
		}
	}

	for li, layer := range p.layers {
		var (
			x     = p.ssx[li]
			delta = ssd[li]
		)
		for j, sw := range layer.ssw {
			for i := range sw {
				sw[i] -= bp.learningRate * delta[j] * x[i]
			}
		}

		var biases = layer.biases
		for j := range biases {
			biases[j] -= bp.learningRate * delta[j] * 1
		}
	}

	return nil
}
