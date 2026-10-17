package goneural

import (
	"github.com/gitchander/neural/neutil/baserw"
)

//------------------------------------------------------------------------------

type LayerConfig struct {
	Neurons int
	AC      ActivationConfig
}

func MakeLayerConfig(neurons int, ac ActivationConfig) LayerConfig {
	return LayerConfig{
		Neurons: neurons,
		AC:      ac,
	}
}

func MakeLayerConfigAF(neurons int, af ActivationFunc) LayerConfig {
	return LayerConfig{
		Neurons: neurons,
		AC: ActivationConfig{
			AF: af,
		},
	}
}

func MakeLayerConfigSoftmax(neurons int) LayerConfig {
	return LayerConfig{
		Neurons: neurons,
		AC: ActivationConfig{
			IsSoftmax: true,
		},
	}
}

//------------------------------------------------------------------------------

func readLayerConfig(br *baserw.BaseReader) (*LayerConfig, error) {
	neurons, err := br.ReadCompactInt()
	if err != nil {
		return nil, err
	}
	pac, err := readActivationConfig(br)
	if err != nil {
		return nil, err
	}
	lc := &LayerConfig{
		Neurons: neurons,
		AC:      *pac,
	}
	return lc, nil
}

func writeLayerConfig(bw *baserw.BaseWriter, lc LayerConfig) error {

	var err error

	err = bw.WriteCompactInt(lc.Neurons)
	if err != nil {
		return err
	}

	err = writeActivationConfig(bw, lc.AC)
	if err != nil {
		return err
	}

	return nil
}

//------------------------------------------------------------------------------

type layer struct {
	ac ActivationConfig

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

	p := &layer{
		ac:      lc.AC,
		neurons: neurons,
		outputs: make([]float64, lc.Neurons),
	}

	return p, nil
}

func (lr *layer) getLayerConfig() LayerConfig {
	return LayerConfig{
		Neurons: len(lr.neurons),
		AC:      lr.ac,
	}
}

func (lr *layer) forward(inputs []float64) {

	// Calculate only logits
	for j, currNeuron := range lr.neurons {
		lr.outputs[j] = currNeuron.calcLogit(inputs) // logit
	}

	lr.ac.Activate(lr.outputs)
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
