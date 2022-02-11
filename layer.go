package neural

type Layer struct {
	ActivationType ActivationType
	Neurons        int
}

type layer struct {
	at      ActivationType
	actFunc ActivationFunc
	neurons []*neuron
}

func MakeLayers(at ActivationType, ds ...int) []Layer {
	layers := make([]Layer, len(ds))
	for i, d := range ds {
		layers[i] = Layer{
			ActivationType: at,
			Neurons:        d,
		}
	}
	return layers
}
