package goneural

type layer struct {
	lc LayerConfig

	afe *actFuncExt

	neurons []*neuron
	outputs []float64
}

func newLayer(lc LayerConfig, weightsPerNeuron int) (*layer, error) {
	neurons := make([]*neuron, lc.Neurons)
	for j := range neurons {
		n := new(neuron)
		n.weights = make([]float64, weightsPerNeuron)
		neurons[j] = n
	}
	afe, err := makeActivationFunc(lc.Activation)
	if err != nil {
		return nil, err
	}

	p := &layer{
		lc:      lc,
		afe:     afe,
		neurons: neurons,
		outputs: make([]float64, lc.Neurons),
	}

	return p, nil
}

func (lr *layer) forward(inputs []float64) {

	// Calculate only logits
	for j, currNeuron := range lr.neurons {
		lr.outputs[j] = currNeuron.calcLogit(inputs) // logit
	}

	// If current activation function is Softmax
	if lr.afe.isSoftmax {
		softmax(lr.outputs, lr.outputs)
	} else {
		for i := range lr.outputs {
			lr.outputs[i] = lr.afe.af.Func(lr.outputs[i])
		}
	}
}

func (lr *layer) updateWeights(inputs []float64, currDeltas []float64, learningRate float64) {
	for j, currNeuron := range lr.neurons {
		for i := range inputs {
			currNeuron.weights[i] -= learningRate * currDeltas[j] * inputs[i]
		}
		currNeuron.bias -= learningRate * currDeltas[j] * 1
	}
}

//------------------------------------------------------------------------------

func MakeLayers(activationName string, ds ...int) []LayerConfig {
	ac := ActivationConfig{
		Name: activationName,
	}
	layers := make([]LayerConfig, len(ds))
	for i, d := range ds {
		layers[i] = LayerConfig{
			Activation: ac,
			Neurons:    d,
		}
	}
	return layers
}
