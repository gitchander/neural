package goneural

type neuron struct {
	weights []float64 // input weights
	bias    float64   // input bias
}

func (n *neuron) calcLogit(inputs []float64) float64 {
	sum := n.bias
	for i, input := range inputs {
		sum += input * n.weights[i]
	}
	return sum
}

func (n *neuron) Activate(inputs []float64, af ActivationFunc) float64 {
	logit := n.calcLogit(inputs)
	return af.Func(logit)
}
