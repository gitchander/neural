package goneural

import (
	"github.com/gitchander/neural/neutil/baserw"
)

// ActivationType, ActivationFunc
// ActivationName string

type ActivationConfig struct {
	Name   string
	Params []float64
}

type LayerConfig struct {
	Activation ActivationConfig
	Neurons    int // Number of neurons per layer
}

func writeLayerConfig(bw *baserw.BaseWriter, lc LayerConfig) error {

	var err error

	err = bw.WriteString(lc.Activation.Name)
	if err != nil {
		return err
	}

	err = bw.WriteFloat64s(lc.Activation.Params)
	if err != nil {
		return err
	}

	err = bw.WriteCompactInt(lc.Neurons)
	if err != nil {
		return err
	}

	return nil
}

func readLayerConfig(br *baserw.BaseReader) (*LayerConfig, error) {

	name, err := br.ReadString()
	if err != nil {
		return nil, err
	}

	params, err := br.ReadFloat64s()
	if err != nil {
		return nil, err
	}

	neurons, err := br.ReadCompactInt()
	if err != nil {
		return nil, err
	}

	lc := &LayerConfig{
		Activation: ActivationConfig{
			Name:   name,
			Params: params,
		},
		Neurons: neurons,
	}

	return lc, nil
}
