package neural

import (
	"errors"
	"fmt"
	"math/rand"
)

/*
______
\\    |
 \\
 /
/_____|

*/

type neuron struct {
	weights []float64
	bias    float64
	delta   float64 // for backpropagation
	out     float64 // output value
}

type layer struct {
	ns []*neuron
}

// Multilayer Perceptron
type Perceptron struct {
	a      float64
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
		a:      0.5,
		layers: layers,
	}
}

func (p *Perceptron) RandomizeWeights(r *rand.Rand) {
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

func (p *Perceptron) SetInputs(inputs []float64) error {

	if len(p.layers) == 0 {
		return errors.New("network is not init")
	}

	ns := p.layers[0].ns

	if len(inputs) != len(ns) {
		return fmt.Errorf("count inputs (%d) not equal count network inputs (%d)", len(inputs), len(ns))
	}

	for i, v := range inputs {
		ns[i].out = v
	}

	return nil
}

func (p *Perceptron) GetOutputs(outputs []float64) error {

	if len(p.layers) == 0 {
		return errors.New("network is not init")
	}

	ns := p.layers[len(p.layers)-1].ns

	if len(outputs) != len(ns) {
		return fmt.Errorf("count outputs (%d) not equal count network outputs (%d)", len(outputs), len(ns))
	}

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
			sum += n.bias
			n.out = sigmoid(sum, p.a)
		}
	}
}

func (p *Perceptron) CalculateMSE(sample Sample) float64 {
	p.SetInputs(sample.Inputs)
	p.Calculate()
	//p.GetOutputs(outputs)

	last := p.layers[len(p.layers)-1]
	var sum float64
	for i, n := range last.ns {
		delta := sample.Outputs[i] - n.out
		sum += delta * delta
	}
	return sum / float64(len(last.ns))
}

func (p *Perceptron) PrintWeights() {
	for k, layer := range p.layers {
		fmt.Printf("layer %d:\n", k)
		for j, n := range layer.ns {
			for i, w := range n.weights {
				fmt.Printf("%d->%d: %.7f\n", i, j, w)
			}
		}
	}
}

func (p *Perceptron) PrintBiases() {
	for k, layer := range p.layers {
		fmt.Printf("layer %d:\n", k)
		for j, n := range layer.ns {
			fmt.Printf("->%d: %.7f\n", j, n.bias)
		}
	}
}
