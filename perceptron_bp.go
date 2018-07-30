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
		speed: 0.5,
	}
}

// outputs - ideal outputs
func (bp *Backpropagation) Learn(inputs, outputs []float64) error {

	p := bp.p
	a := p.a

	err := p.SetInputs(inputs)
	if err != nil {
		return err
	}
	p.Calculate()

	ssd := bp.ssd
	m := len(ssd) - 1

	var (
		x     = p.ssx[m+1]
		delta = ssd[m]
	)
	for j := range delta {
		delta[j] = sigmoidPrime(x[j], a) * (x[j] - outputs[j])
	}
	m--

	for ; m >= 0; m-- {
		var (
			x     = p.ssx[m+1]
			delta = ssd[m]

			deltaChildren   = ssd[m+1]
			weightsChildren = p.sssw[m+1]
		)
		for j := range delta {
			var sum float64
			for k := range deltaChildren {
				sum += deltaChildren[k] * weightsChildren[k][j]
			}
			delta[j] = sigmoidPrime(x[j], a) * sum
		}
	}

	for m, ssw := range p.sssw {
		var (
			x     = p.ssx[m]
			delta = ssd[m]
		)
		for j, sw := range ssw {
			for i := range sw {
				sw[i] -= bp.speed * delta[j] * x[i]
			}
		}
	}

	return nil
}
