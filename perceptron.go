package neural

import (
	"errors"
	"fmt"
	"math/rand"
)

type layer struct {
	ssw    [][]float64 // synapse weights
	biases []float64
}

// Multilayer Perceptron
type Perceptron struct {
	a      float64
	ssx    [][]float64 // neuron outputs
	layers []layer
}

func NewPerceptron(ds ...int) *Perceptron {

	ssx := make([][]float64, len(ds))
	for i := range ssx {
		ssx[i] = make([]float64, ds[i])
	}

	layers := make([]layer, len(ds)-1)
	for i := range layers {
		layers[i] = layer{
			ssw:    newMatrix2(ds[i+1], ds[i]),
			biases: make([]float64, ds[i+1]),
		}
	}

	return &Perceptron{
		a:      0.5,
		ssx:    ssx,
		layers: layers,
	}
}

func (p *Perceptron) RandomizeWeights(r *rand.Rand) {
	for _, layer := range p.layers {
		for _, sw := range layer.ssw {
			for i := range sw {
				sw[i] = randWeight(r)
			}
		}
		biases := layer.biases
		for i := range biases {
			biases[i] = randWeight(r)
		}
	}
}

func (p *Perceptron) SetInputs(inputs []float64) error {

	if len(p.ssx) == 0 {
		return errors.New("network is not init")
	}

	inputLayer := p.ssx[0]

	if len(inputs) != len(inputLayer) {
		return fmt.Errorf("count inputs (%d) not equal count network inputs (%d)", len(inputs), len(inputLayer))
	}

	for i, v := range inputs {
		inputLayer[i] = v
	}

	return nil
}

func (p *Perceptron) GetOutputs(outputs []float64) error {

	if len(p.ssx) == 0 {
		return errors.New("network is not init")
	}

	outputLayer := p.ssx[len(p.ssx)-1]

	if len(outputs) != len(outputLayer) {
		return fmt.Errorf("count outputs (%d) not equal count network outputs (%d)", len(outputs), len(outputLayer))
	}

	for i, v := range outputLayer {
		outputs[i] = v
	}

	return nil
}

func (p *Perceptron) Calculate() {
	for k, layer := range p.layers {
		var (
			sxi = p.ssx[k]
			sxj = p.ssx[k+1]

			biases = layer.biases
		)
		for j, sw := range layer.ssw {
			sum := 0.0
			for i, w := range sw {
				sum += w * sxi[i]
			}
			sum += biases[j]
			sxj[j] = sigmoid(sum, p.a)
		}
	}
}

func (p *Perceptron) CalculateMSE(sample Sample) float64 {
	p.SetInputs(sample.Inputs)
	p.Calculate()
	//p.GetOutputs(outputs)
	outputs := p.ssx[len(p.ssx)-1]
	return MSE(outputs, sample.Outputs)
}

func (p *Perceptron) PrintWeights() {
	for l, layer := range p.layers {
		fmt.Printf("layer %d:\n", l)
		for j, sw := range layer.ssw {
			for i, w := range sw {
				fmt.Printf("%d->%d: %.7f\n", i, j, w)
			}
		}
	}
}

func (p *Perceptron) PrintBiases() {
	for _, layer := range p.layers {
		for j, b := range layer.biases {
			fmt.Printf("->%d: %.7f\n", j, b)
		}
	}
}
