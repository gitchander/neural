package goneural

type layer struct {
	lc LayerConfig

	afe *actFuncExt

	neurons []*neuron
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
	}

	return p, nil
}

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
