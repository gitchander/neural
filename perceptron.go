package neural

import (
	"errors"
	"math/rand"

	"github.com/gitchander/neural/neutil/random"
)

type neuron struct {
	weights []float64 // input weights
	bias    float64   // input bias
	out     float64   // output value

	delta float64 // for backpropagation
}

type layer struct {
	actFunc ActivationFunc
	neurons []*neuron
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
		neurons := make([]*neuron, ds[i])
		for j := range neurons {
			n := new(neuron)
			if i > 0 {
				n.weights = make([]float64, ds[i-1])
			}
			neurons[j] = n
		}
		layers[i] = &layer{
			actFunc: new(Sigmoid),
			neurons: neurons,
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
	for i, l := range p.layers {
		ds[i] = len(l.neurons)
	}
	return ds
}

func (p *MLP) RandomizeWeights() {
	r := random.NewRandNow()
	p.RandomizeWeightsRand(r)
}

func (p *MLP) RandomizeWeightsRand(r *rand.Rand) {
	for k := 1; k < len(p.layers); k++ {
		l := p.layers[k]
		for _, n := range l.neurons {
			ws := n.weights
			for i := range ws {
				ws[i] = randWeight(r)
			}
			n.bias = randWeight(r)
		}
	}
}

func (p *MLP) SetInputs(inputs []float64) {
	for i, n := range p.inputLayer.neurons {
		n.out = inputs[i]
	}
}

func (p *MLP) GetOutputs(outputs []float64) {
	for i, n := range p.outputLayer.neurons {
		outputs[i] = n.out
	}
}

func (p *MLP) Calculate() {
	for k := 1; k < len(p.layers); k++ {
		var (
			prevLayer = p.layers[k-1]
			currLayer = p.layers[k]
		)
		for _, currNeuron := range currLayer.neurons {
			var sum float64
			for i, prevNeuron := range prevLayer.neurons {
				sum += currNeuron.weights[i] * prevNeuron.out
			}
			sum += currNeuron.bias * 1
			currNeuron.out = currLayer.actFunc.Func(sum)
		}
	}
}
