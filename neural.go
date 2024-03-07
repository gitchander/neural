package neural

import (
	"errors"
	"math/rand"

	"github.com/gitchander/neural/neutil/random"
)

// Feedforward neural network
type Neural struct {
	layers []*layer
}

func NewNeural(rs []LayerInfo) (*Neural, error) {
	if len(rs) < 2 {
		return nil, errors.New("neural: invalid number of layers")
	}
	layers := make([]*layer, len(rs))
	for i, r := range rs {
		neurons := make([]*neuron, r.Neurons)
		for j := range neurons {
			n := new(neuron)
			if i > 0 {
				n.weights = make([]float64, rs[i-1].Neurons)
			}
			neurons[j] = n
		}
		layers[i] = &layer{
			at:      r.ActivationType,
			actFunc: MakeActivationFunc(r.ActivationType),
			neurons: neurons,
		}
	}
	p := &Neural{
		layers: layers,
	}
	return p, nil
}

func (p *Neural) getInputLayer() *layer {
	return p.layers[0]
}

func (p *Neural) getOutputLayer() *layer {
	return p.layers[len(p.layers)-1]
}

func (p *Neural) Topology() []int {
	ds := make([]int, len(p.layers))
	for i, l := range p.layers {
		ds[i] = len(l.neurons)
	}
	return ds
}

func (p *Neural) RandomizeWeights() {
	r := random.NewRandSeed(random.NextSeed())
	p.randomizeWeightsRand(r)
}

func (p *Neural) randomizeWeightsRand(r *rand.Rand) {
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

func (p *Neural) SetInputs(inputs []float64) {
	ns := p.getInputLayer().neurons
	for i, n := range ns {
		n.out = inputs[i]
	}
}

func (p *Neural) GetOutputs(outputs []float64) {
	ns := p.getOutputLayer().neurons
	for i, n := range ns {
		outputs[i] = n.out
	}
}

// forward
func (p *Neural) Calculate() {
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

		// If current activation function is Softmax
		if currLayer.at == ActSoftmax {
			var sum float64
			for _, currNeuron := range currLayer.neurons {
				sum += currNeuron.out
			}
			for _, currNeuron := range currLayer.neurons {
				currNeuron.out /= sum
			}
		}
	}
}
