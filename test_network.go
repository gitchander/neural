package neural

import (
	"fmt"
	"math/rand"
)

type neuron struct {
	inputs  []*synapse
	outputs []*synapse

	val float64
}

type synapse struct {
	input  *neuron
	output *neuron

	weight float64
}

func joinNeuronsBySynapse(input, output *neuron) {
	syn := &synapse{
		input:  input,
		output: output,
	}
	input.outputs = append(input.outputs, syn)
	output.inputs = append(output.inputs, syn)
}

type testPerceptron struct {
	layers [][]*neuron
}

func newTestPerceptron(ds ...int) *testPerceptron {

	n := len(ds)
	layers := make([][]*neuron, n)
	for i := range layers {
		layers[i] = make([]*neuron, ds[i])
		for j := range layers[i] {
			layers[i][j] = new(neuron)
		}
	}

	for i := 0; i < n-1; i++ {
		var (
			layer0 = layers[i+0]
			layer1 = layers[i+1]
		)
		for _, n0 := range layer0 {
			for _, n1 := range layer1 {
				joinNeuronsBySynapse(n0, n1)
			}
		}
	}

	return &testPerceptron{
		layers: layers,
	}
}

func (p *testPerceptron) RandomizeWeights(r *rand.Rand) {
	for i := 1; i < len(p.layers); i++ {
		layer := p.layers[i]
		for _, n := range layer {
			for _, syn := range n.inputs {
				syn.weight = randWeight(r)
			}
		}
	}
}

func (p *testPerceptron) SetInputs(inputs []float64) {
	layer := p.layers[0] // first layer
	for i := range layer {
		layer[i].val = inputs[i]
	}
}

func (p *testPerceptron) GetOutputs(outputs []float64) {
	layer := p.layers[len(p.layers)-1] // last layer
	for i := range layer {
		outputs[i] = layer[i].val
	}
}

func (p *testPerceptron) Calculate() {
	for i := 1; i < len(p.layers); i++ {
		layer := p.layers[i]
		for _, n := range layer {
			sum := 0.0
			for _, syn := range n.inputs {
				sum += syn.input.val * syn.weight
			}
			n.val = sigmoid(sum, 1)
		}
	}
}

func (p *testPerceptron) PrintWeights() {
	for i := 1; i < len(p.layers); i++ {

		fmt.Printf("layer %d:\n", i-1)

		layer := p.layers[i]
		for to, ns := range layer {
			for from, syn := range ns.inputs {
				fmt.Printf("%d->%d: %.7f\n", from, to, syn.weight)
			}
		}
	}
}

func testBackpropagation(p *testPerceptron, inputs, outputs []float64) {
	p.SetInputs(inputs)
	p.Calculate()

	a := 1.0
	speed := 0.5

	//	//	ssd := bp.ssd
	//	//	m := len(ssd) - 1

	//	//	var (
	//	//		x     = n.ssx[m+1]
	//	//		delta = ssd[m]
	//	//	)

	ssd := make([][]float64, len(p.layers)-1)
	for i := range ssd {
		ssd[i] = make([]float64, len(p.layers[i+1]))
	}

	m := len(ssd) - 1
	delta := ssd[m]
	layer := p.layers[m+1] // last layer

	for j, n := range layer {
		delta[j] = sigmoidPrime(n.val, a) * (n.val - outputs[j])
	}
	m--

	for ; m >= 0; m-- {
		var (
			delta         = ssd[m]
			deltaChildren = ssd[m+1]
			layer         = p.layers[m+1]
		)
		for j, n := range layer {
			sum := 0.0
			for k, syn := range n.outputs {
				sum += deltaChildren[k] * syn.weight
			}
			delta[j] = sigmoidPrime(n.val, a) * sum
		}
	}

	for m, sd := range ssd {
		var (
			//x     = n.ssx[m]
			//delta = ssd[m]
			layer = p.layers[m+1]
		)
		for j, d := range sd {
			n := layer[j]

			for _, syn := range n.inputs {
				syn.weight -= speed * d * syn.input.val
			}
		}
	}

}
