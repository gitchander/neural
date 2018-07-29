package neural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// Perceptron
type Network struct {
	a    float64
	ssx  [][]float64
	sssw [][][]float64 // weights
}

func NewNetwork(ds ...int) *Network {

	ssx := make([][]float64, len(ds))
	for i := range ssx {
		ssx[i] = make([]float64, ds[i])
	}

	sssw := make([][][]float64, len(ds)-1)
	for i := range sssw {
		sssw[i] = newMatrix2(ds[i+1], ds[i])
	}

	return &Network{
		a:    1.5,
		ssx:  ssx,
		sssw: sssw,
	}
}

func newMatrix2(n, m int) [][]float64 {
	ssv := make([][]float64, n)
	for i := range ssv {
		ssv[i] = make([]float64, m)
	}
	return ssv
}

func (n *Network) RandomizeWeights(r *rand.Rand) {
	for _, ssw := range n.sssw {
		for _, sw := range ssw {
			for i := range sw {
				sw[i] = 2 * (r.Float64() - 0.5)
			}
		}
	}
}

func (n *Network) SetInputs(inputs []float64) error {

	if len(n.ssx) == 0 {
		return errors.New("network is not init")
	}

	inputLayer := n.ssx[0]

	if len(inputs) != len(inputLayer) {
		return fmt.Errorf("count inputs (%d) not equal count network inputs (%d)", len(inputs), len(inputLayer))
	}

	for i, v := range inputs {
		inputLayer[i] = v
	}

	return nil
}

func (n *Network) GetOutputs(outputs []float64) error {

	if len(n.ssx) == 0 {
		return errors.New("network is not init")
	}

	outputLayer := n.ssx[len(n.ssx)-1]

	if len(outputs) != len(outputLayer) {
		return fmt.Errorf("count outputs (%d) not equal count network outputs (%d)", len(outputs), len(outputLayer))
	}

	for i, v := range outputLayer {
		outputs[i] = v
	}

	return nil
}

func (n *Network) Calculate() {
	for k, ssw := range n.sssw {
		var (
			sxi = n.ssx[k]
			sxj = n.ssx[k+1]
		)
		for j, sw := range ssw {
			sum := 0.0
			for i, w := range sw {
				sum += w * sxi[i]
			}
			sxj[j] = sigmoid(sum, n.a)
		}
	}
}

func sigmoid(x, a float64) float64 {
	return 1 / (1 + math.Exp(-2*a*x))
}

func sigmoidPrime(x, a float64) float64 {
	return 2 * a * x * (1 - x)
}

type Backpropagation struct {
	n     *Network
	ssd   [][]float64 // delta Weight
	speed float64     // 0 < speed < 1
}

func NewBackpropagation(n *Network) *Backpropagation {

	ssd := make([][]float64, len(n.sssw))
	for k, ssw := range n.sssw {
		ssd[k] = make([]float64, len(ssw))
	}

	return &Backpropagation{
		n:     n,
		ssd:   ssd,
		speed: 0.3,
	}
}

// outputs - ideal outputs
func (bp *Backpropagation) Learn(inputs, outputs []float64) error {

	n := bp.n
	a := n.a

	err := n.SetInputs(inputs)
	if err != nil {
		return err
	}
	n.Calculate()

	ssd := bp.ssd
	m := len(ssd) - 1

	var (
		x     = n.ssx[m+1]
		delta = ssd[m]
	)
	for j := range delta {
		delta[j] = sigmoidPrime(x[j], a) * (x[j] - outputs[j])
	}
	m--

	for ; m >= 0; m-- {
		var (
			x     = n.ssx[m+1]
			delta = ssd[m]

			deltaChildren   = ssd[m+1]
			weightsChildren = n.sssw[m+1]
		)
		for j := range delta {
			sum := 0.0
			for k := range deltaChildren {
				sum += deltaChildren[k] * weightsChildren[k][j]
			}
			delta[j] = sigmoidPrime(x[j], a) * sum
		}
	}

	for m, ssw := range n.sssw {
		var (
			x     = n.ssx[m]
			delta = ssd[m]
		)
		for j, sw := range ssw {
			for i := range sw {
				sw[i] -= bp.speed * delta[j] * x[i]
			}
		}
	}

	return nil
}
