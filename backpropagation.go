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
	ssd := make([][]float64, len(p.sssw))
	for k, ssw := range p.sssw {
		ssd[k] = make([]float64, len(ssw))
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

	for layer := last - 1; layer >= 0; layer-- {
		var (
			x     = p.ssx[layer+1]
			delta = ssd[layer]

			deltaNext   = ssd[layer+1]
			weightsNext = p.sssw[layer+1]
		)

		for j := range delta {
			var sum float64
			for k := range deltaNext {
				sum += deltaNext[k] * weightsNext[k][j]
			}
			delta[j] = sigmoidPrime(x[j], p.a) * sum
		}
	}

	for layer, ssw := range p.sssw {
		var (
			x     = p.ssx[layer]
			delta = ssd[layer]
		)
		for j, sw := range ssw {
			for i := range sw {
				sw[i] -= bp.learningRate * delta[j] * x[i]
			}
		}

		var bias = p.biases[layer]
		for j := range bias {
			bias[j] -= bp.learningRate * delta[j]
		}
	}

	return nil
}
