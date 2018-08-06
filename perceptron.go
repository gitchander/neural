package neural

import (
	"errors"
	"fmt"
	"math/rand"
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

// Multilayer Perceptron
// FeedForward
type Perceptron struct {
	af     ActivationFunc
	layers []*layer
}

func NewPerceptron(ds ...int) *Perceptron {
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
	return &Perceptron{
		af:     Sigmoid{},
		layers: layers,
	}
}

func (p *Perceptron) RandomizeWeights(r *rand.Rand) {
	for k := 1; k < len(p.layers); k++ {
		layer := p.layers[k]
		for _, n := range layer.ns {
			ws := n.weights
			for i := range ws {
				ws[i] = randRange(r, -0.5, 0.5)
			}
			n.bias = randRange(r, -0.5, 0.5)
		}
	}
}

func (p *Perceptron) checkInputs(inputs []float64) error {
	if len(p.layers) == 0 {
		return errors.New("network is not init")
	}
	firstLayer := p.layers[0]
	if len(inputs) != len(firstLayer.ns) {
		return fmt.Errorf("count inputs (%d) not equal count network inputs (%d)",
			len(inputs), len(firstLayer.ns))
	}
	return nil
}

func (p *Perceptron) checkOutputs(outputs []float64) error {
	if len(p.layers) == 0 {
		return errors.New("network is not init")
	}
	ns := p.layers[len(p.layers)-1].ns
	if len(outputs) != len(ns) {
		return fmt.Errorf("count outputs (%d) not equal count network outputs (%d)",
			len(outputs), len(ns))
	}
	return nil
}

func (p *Perceptron) SetInputs(inputs []float64) error {
	if err := p.checkInputs(inputs); err != nil {
		return err
	}
	ns := p.layers[0].ns
	for i, v := range inputs {
		ns[i].out = v
	}
	return nil
}

func (p *Perceptron) GetOutputs(outputs []float64) error {
	if err := p.checkOutputs(outputs); err != nil {
		return err
	}
	ns := p.layers[len(p.layers)-1].ns
	for i, n := range ns {
		outputs[i] = n.out
	}
	return nil
}

func (p *Perceptron) Calculate() {
	for k := 1; k < len(p.layers); k++ {
		var (
			layer     = p.layers[k]
			layerPrev = p.layers[k-1]
		)
		for _, n := range layer.ns {
			var sum float64
			for i, n_prev := range layerPrev.ns {
				sum += n.weights[i] * n_prev.out
			}
			sum += n.bias * 1
			n.out = p.af.Func(sum)
		}
	}
}

func (p *Perceptron) CalculateMSE(sample Sample) float64 {
	err := p.SetInputs(sample.Inputs)
	if err != nil {
		panic(err)
	}

	p.Calculate()

	err = p.checkOutputs(sample.Outputs)
	if err != nil {
		panic(err)
	}

	ns := p.layers[len(p.layers)-1].ns
	var sum float64
	for i, n := range ns {
		delta := sample.Outputs[i] - n.out
		sum += delta * delta
	}
	return sum / float64(len(ns))
}
