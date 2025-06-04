package goneural

type layer struct {
	lc LayerConfig

	actFunc ActivationFunc
	neurons []*neuron
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
