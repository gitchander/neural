package neural

type LayerInfo struct {
	ActivationType ActivationType
	Neurons        int
}

type layer struct {
	at      ActivationType
	actFunc ActivationFunc
	neurons []*neuron
}

func MakeLayers(at ActivationType, ds ...int) []LayerInfo {
	layers := make([]LayerInfo, len(ds))
	for i, d := range ds {
		layers[i] = LayerInfo{
			ActivationType: at,
			Neurons:        d,
		}
	}
	return layers
}
