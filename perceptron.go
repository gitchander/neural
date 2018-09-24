package neural

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

type neuron struct {
	weights []float64 // InputWeights
	bias    float64
	delta   float64 // for backpropagation
	out     float64 // output value
}

type layer struct {
	ns []*neuron
}

// Multilayer perceptron (MLP)
// FeedForward
// Fully Connected Layers
type MLP struct {
	actFunc ActivationFunc
	layers  []*layer

	inputLayer  *layer
	outputLayer *layer
}

func NewMLP(ds ...int) (*MLP, error) {
	if len(ds) < 2 {
		return nil, errors.New("neural: invalid number of layers")
	}
	layers := make([]*layer, len(ds))
	for i := range layers {
		ns := make([]*neuron, ds[i])
		for j := range ns {
			n := new(neuron)
			if i > 0 {
				n.weights = make([]float64, ds[i-1])
			}
			ns[j] = n
		}
		layers[i] = &layer{ns: ns}
	}
	p := &MLP{
		actFunc:     Sigmoid{},
		layers:      layers,
		inputLayer:  layers[0],
		outputLayer: layers[len(layers)-1],
	}
	return p, nil
}

func (p *MLP) Topology() []int {
	ds := make([]int, len(p.layers))
	for i, layer := range p.layers {
		ds[i] = len(layer.ns)
	}
	return ds
}

func (p *MLP) SetActivationFunc(af ActivationFunc) {
	p.actFunc = af
}

func (p *MLP) RandomizeWeights() {
	r := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	p.RandomizeWeightsRand(r)
}

func (p *MLP) RandomizeWeightsRand(r *rand.Rand) {
	for k := 1; k < len(p.layers); k++ {
		layer := p.layers[k]
		for _, n := range layer.ns {
			ws := n.weights
			for i := range ws {
				ws[i] = randWeight(r)
			}
			n.bias = randWeight(r)
		}
	}
}

func (p *MLP) checkInputs(inputs []float64) error {
	ns := p.inputLayer.ns
	if len(inputs) != len(ns) {
		return fmt.Errorf("count inputs (%d) not equal count network inputs (%d)",
			len(inputs), len(ns))
	}
	return nil
}

func (p *MLP) checkOutputs(outputs []float64) error {
	ns := p.outputLayer.ns
	if len(outputs) != len(ns) {
		return fmt.Errorf("count outputs (%d) not equal count network outputs (%d)",
			len(outputs), len(ns))
	}
	return nil
}

func (p *MLP) SetInputs(inputs []float64) error {
	if err := p.checkInputs(inputs); err != nil {
		return err
	}
	ns := p.inputLayer.ns
	for i, n := range ns {
		n.out = inputs[i]
	}
	return nil
}

func (p *MLP) GetOutputs(outputs []float64) error {
	if err := p.checkOutputs(outputs); err != nil {
		return err
	}
	ns := p.outputLayer.ns
	for i, n := range ns {
		outputs[i] = n.out
	}
	return nil
}

func (p *MLP) Calculate() {
	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer = p.layers[k-1]
			layer     = p.layers[k]
		)
		for _, n := range layer.ns {
			var sum float64
			for i, n_prev := range prevLayer.ns {
				sum += n.weights[i] * n_prev.out
			}
			sum += n.bias * 1
			n.out = p.actFunc.Func(sum)
		}
	}
}
