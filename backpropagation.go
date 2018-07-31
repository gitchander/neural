package neural

type Backpropagation struct {
	p     *Perceptron
	ssd   [][]float64 // delta Weight
	speed float64     // 0 < speed < 1
}

func NewBackpropagation(p *Perceptron) *Backpropagation {

	ssd := make([][]float64, len(p.sssw))
	for k, ssw := range p.sssw {
		ssd[k] = make([]float64, len(ssw))
	}

	return &Backpropagation{
		p:     p,
		ssd:   ssd,
		speed: 0.7,
	}
}

func (bp *Backpropagation) SetSpeed(speed float64) {
	bp.speed = speed
}

// outputs - ideal outputs
func (bp *Backpropagation) Learn(inputs, outputs []float64) error {

	p := bp.p

	err := p.SetInputs(inputs)
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
		delta[j] = sigmoidPrime(x[j], p.a) * (x[j] - outputs[j])
	}

	for curr := last - 1; curr >= 0; curr-- {
		var (
			x     = p.ssx[curr+1]
			delta = ssd[curr]

			deltaChildren   = ssd[curr+1]
			weightsChildren = p.sssw[curr+1]
		)

		for j := range delta {
			var sum float64
			for k := range deltaChildren {
				sum += deltaChildren[k] * weightsChildren[k][j]
			}
			delta[j] = sigmoidPrime(x[j], p.a) * sum
		}
	}

	for curr, ssw := range p.sssw {
		var (
			x     = p.ssx[curr]
			delta = ssd[curr]
		)
		for j, sw := range ssw {
			for i := range sw {
				sw[i] -= bp.speed * delta[j] * x[i]
			}
		}
	}

	return nil
}
