package neural

import (
	"errors"
	"fmt"
	"math/rand"
)

type Perceptron struct {
	a    float64
	ssx  [][]float64   // neuron outputs
	sssw [][][]float64 // synapse weights
}

func NewPerceptron(ds ...int) *Perceptron {

	ssx := make([][]float64, len(ds))
	for i := range ssx {
		ssx[i] = make([]float64, ds[i])
	}

	sssw := make([][][]float64, len(ds)-1)
	for i := range sssw {
		sssw[i] = newMatrix2(ds[i+1], ds[i])
	}

	return &Perceptron{
		a:    0.5,
		ssx:  ssx,
		sssw: sssw,
	}
}

func (p *Perceptron) RandomizeWeights(r *rand.Rand) {
	for _, ssw := range p.sssw {
		for _, sw := range ssw {
			for i := range sw {
				sw[i] = randWeight(r)
			}
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
	for k, ssw := range p.sssw {
		var (
			sxi = p.ssx[k]
			sxj = p.ssx[k+1]
		)
		for j, sw := range ssw {
			sum := 0.0
			for i, w := range sw {
				sum += w * sxi[i]
			}
			sxj[j] = sigmoid(sum, p.a)
		}
	}
}

func (p *Perceptron) PrintWeights() {
	for l, ssw := range p.sssw {
		fmt.Printf("layer %d:\n", l)
		for j, sw := range ssw {
			for i, w := range sw {
				fmt.Printf("%d->%d: %.7f\n", i, j, w)
			}
		}
	}
}
