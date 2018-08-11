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

func crop_01(x float64) float64 {
	if x < 0 {
		x = 0
	}
	if x > 1 {
		x = 1
	}
	return x
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
		n.delta = p.af.Derivative(n.out) * errFunc.Derivative(sample.Outputs[j], n.out)
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

func (bp *Backpropagation) LearnSamples(samples []Sample) (le float64, err error) {
	var sum float64
	for _, sample := range samples {
		err = bp.Learn(sample)
		if err != nil {
			return 0, err
		}
		sum += bp.p.SampleError(sample)
	}
	le = sum / float64(len(samples))
	return le, nil
}
