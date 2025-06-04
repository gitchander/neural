package goneural

type neuron struct {
	weights []float64 // input weights
	bias    float64   // input bias

	out float64 // output value
}
