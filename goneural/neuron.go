package goneural

type neuron struct {
	weights []float64 // input weights
	bias    float64   // input bias

	out float64 // output value
}

// func (n *neuron) Activate(inputs []float64, af ActivationFunc) float64 {
// 	var sum float64
// 	sum = n.bias
// 	for i, input := range inputs {
// 		sum += input * n.weights[i]
// 	}
// 	n.out = af.Func(sum)
// 	return n.out
// }

// func (n *neuron) Activate(inputs []float64) float64 {
// 	var sum float64
// 	sum = n.bias
// 	for i, input := range inputs {
// 		sum += input * n.weights[i]
// 	}
// 	n.out = sum
// 	return n.out
// }
