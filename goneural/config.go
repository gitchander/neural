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
	Neurons    int
}

func writeFloat64s(bw *baserw.BaseWriter, vs []float64) error {
	err := bw.WriteCompactInt(len(vs))
	if err != nil {
		return err
	}
	for _, v := range vs {
		err = bw.WriteFloat64(v)
		if err != nil {
			return err
		}
	}
	return nil
}

func readFloat64s(br *baserw.BaseReader) ([]float64, error) {
	n, err := br.ReadCompactInt()
	if err != nil {
		return nil, err
	}
	vs := make([]float64, n)
	for i := range vs {
		v, err := br.ReadFloat64()
		if err != nil {
			return nil, err
		}
		vs[i] = v
	}
	return vs, nil
}

func writeLayerConfig(bw *baserw.BaseWriter, lc LayerConfig) error {

	var err error

	err = bw.WriteString(lc.Activation.Name)
	if err != nil {
		return err
	}

	err = writeFloat64s(bw, lc.Activation.Params)
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

	params, err := readFloat64s(br)
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
