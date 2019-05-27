package neural

import (
	"errors"
	"math/rand"

	"github.com/gitchander/neural/neutil/random"
)

type neuron struct {
	weights []float64 // input weights
	bias    float64   // input bias
	delta   float64   // for backpropagation
	out     float64   // output value
}

type layer struct {
	actFunc ActivationFunc
	ns      []*neuron
}

// Multilayer perceptron (MLP)
// FeedForward
// Fully Connected Layers
type MLP struct {
	layers []*layer

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
		layers[i] = &layer{
			actFunc: new(Sigmoid),
			ns:      ns,
		}
	}
	p := &MLP{
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

func (p *MLP) RandomizeWeights() {
	r := random.NewRandNow()
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

func (p *MLP) SetInputs(inputs []float64) {
	for i, n := range p.inputLayer.ns {
		n.out = inputs[i]
	}
}

func (p *MLP) GetOutputs(outputs []float64) {
	for i, n := range p.outputLayer.ns {
		outputs[i] = n.out
	}
}

func (p *MLP) Calculate() {
	n := len(p.layers)
	if n == 0 {
		return
	}
	prevLayer := p.layers[0]
	for k := 1; k < n; k++ {
		layer := p.layers[k]
		for _, n := range layer.ns {
			var sum float64
			for i, prevNeuron := range prevLayer.ns {
				sum += n.weights[i] * prevNeuron.out
			}
			sum += n.bias * 1
			n.out = layer.actFunc.Func(sum)
		}
		prevLayer = layer
	}
}
