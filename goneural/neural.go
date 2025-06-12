package goneural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/gitchander/neural/neutil/random"
)

// Feedforward neural network
type Neural struct {
	layers []*layer
}

func NewNeural(lcs []LayerConfig) (*Neural, error) {
	if len(lcs) < 2 {
		return nil, errors.New("neural: invalid number of layers")
	}
	layers := make([]*layer, len(lcs))
	for i, lc := range lcs {

		var weightsPerNeuron int = 0 // for first layer
		if i > 0 {
			weightsPerNeuron = lcs[i-1].Neurons
		}

		l, err := newLayer(lc, weightsPerNeuron)
		if err != nil {
			return nil, fmt.Errorf("make layer[%d]: %w", i, err)
		}

		layers[i] = l
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
			currNeuron.out = currLayer.afe.af.Func(sum)
		}

		// If current activation function is Softmax
		if currLayer.afe.isSoftmax {

			// The Softmax Function

			var sum float64
			for _, currNeuron := range currLayer.neurons {
				currNeuron.out = math.Exp(currNeuron.out)
				sum += currNeuron.out
			}

			// if math.IsInf(sum, +1) {
			// 	fmt.Println("sum:", sum)
			// }

			for _, currNeuron := range currLayer.neurons {
				currNeuron.out /= sum
			}
		}
	}
}

//------------------------------------------------------------------------------

func neuronForward(prevLayer *layer, currNeuron *neuron, actFunc ActivationFunc) {
	var sum float64
	for i, prevNeuron := range prevLayer.neurons {
		sum += currNeuron.weights[i] * prevNeuron.out
	}
	sum += currNeuron.bias * 1
	currNeuron.out = actFunc.Func(sum)
}

func layerForward(prevLayer *layer, currLayer *layer) {
	for _, currNeuron := range currLayer.neurons {
		neuronForward(prevLayer, currNeuron, currLayer.afe.af)
	}
}
