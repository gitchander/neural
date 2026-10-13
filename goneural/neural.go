package goneural

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/gitchander/neural/neutil/random"
)

// Feedforward neural network
// MLP - Multi-Layer Perceptron
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
	inputLayer := p.getInputLayer()
	copy(inputLayer.outputs, inputs)
}

func (p *Neural) GetOutputs(outputs []float64) {
	outputLayer := p.getOutputLayer()
	copy(outputs, outputLayer.outputs)
}

// forward
func (p *Neural) Calculate() {
	for i := 1; i < len(p.layers); i++ {
		var (
			prevLayer = p.layers[i-1]
			currLayer = p.layers[i]
		)
		currLayer.forward(prevLayer.outputs)
	}
}
