package neural

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/gitchander/neural/other"
)

func TestRandWeights(t *testing.T) {

	const randSeed = 100

	type sample struct {
		inputs  []float64
		outputs []float64
	}

	const (
		v0 = 0.1
		v1 = 0.9
	)

	samples := []Sample{
		{Inputs: []float64{v0, v0}, Outputs: []float64{v0}},
		{Inputs: []float64{v1, v0}, Outputs: []float64{v1}},
		{Inputs: []float64{v0, v1}, Outputs: []float64{v1}},
		{Inputs: []float64{v1, v1}, Outputs: []float64{v0}},
	}

	const epochMax = 1000

	var (
		inputs  = make([]float64, 2)
		outputs = make([]float64, 1)
	)

	ds := []int{2, 3, 1}
	r := rand.New(rand.NewSource(randSeed))

	tp := other.NewPerceptron(ds...)
	tp.RandomizeWeights(r)
	//tp.PrintWeights()

	tp.SetInputs(inputs)
	tp.Calculate()
	tp.GetOutputs(outputs)
	fmt.Println(outputs)

	for i := 0; i < epochMax; i++ {
		for _, sample := range samples {
			other.Backpropagation(tp, sample.Inputs, sample.Outputs)
		}
	}
	tp.PrintWeights()

	//	tp.SetInputs(s1[0].inputs)
	//	tp.Calculate()
	//	tp.GetOutputs(outputs)
	//	fmt.Println(outputs)

	fmt.Println("//----------------------------")

	r = rand.New(rand.NewSource(randSeed))
	p := NewPerceptron(ds...)
	p.RandomizeWeights(r)
	//p.PrintWeights()

	p.SetInputs(inputs)
	p.Calculate()
	p.GetOutputs(outputs)
	fmt.Println(outputs)

	bp := NewBackpropagation(p)
	for i := 0; i < epochMax; i++ {
		for _, sample := range samples {
			bp.Learn(sample)
		}
	}
	p.PrintWeights()

	//	p.SetInputs(s1[0].inputs)
	//	p.Calculate()
	//	p.GetOutputs(outputs)
	//	fmt.Println(outputs)
}
