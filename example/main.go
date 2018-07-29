package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/gitchander/neural"
)

func main() {
	mul()
	//xor()
}

func test() {
	n := neural.NewNetwork(2, 3, 1)

	n.RandomizeWeights(newRand())

	inputs := []float64{0.7, 0.0}

	err := n.SetInputs(inputs)
	checkError(err)

	n.Calculate()

	outputs := make([]float64, 1)
	n.GetOutputs(outputs)
	fmt.Println(outputs)

	bp := neural.NewBackpropagation(n)

	for i := 0; i < 1000; i++ {
		bp.Learn(inputs, []float64{0.5})
	}

	err = n.SetInputs(inputs)
	checkError(err)

	n.Calculate()
	n.GetOutputs(outputs)
	fmt.Println(outputs)
}

func mul() {

	const epsilon = 0.001

	n := neural.NewNetwork(2, 3, 2, 1)
	r := newRand()
	n.RandomizeWeights(r)
	bp := neural.NewBackpropagation(n)

	var (
		inputs       = make([]float64, 2)
		outputs      = make([]float64, 1)
		outputsIdeal = make([]float64, 1)
	)

	samplesInEpoch := 1000

	epoch := 0
	epochMax := 10000
	for epoch < epochMax {

		worst := 0.0
		for i := 0; i < samplesInEpoch; i++ {

			a, b := r.Float64(), r.Float64()
			inputs[0] = a
			inputs[1] = b
			outputsIdeal[0] = a * b

			bp.Learn(inputs, outputsIdeal)

			n.SetInputs(inputs)
			n.Calculate()
			n.GetOutputs(outputs)

			mse := neural.MSE(outputs, outputsIdeal)
			if mse > worst {
				worst = mse
			}
		}

		if worst < epsilon {
			break
		}

		epoch++
	}
	if epoch < epochMax {
		fmt.Println("success: epoch =", epoch)
	} else {
		fmt.Println("failure")
	}
}

func xor() {

	const epsilon = 0.001

	const (
		v0 = 0.1
		v1 = 0.9
	)
	samples := []struct {
		inputs  []float64
		outputs []float64
	}{
		{
			inputs:  []float64{v0, v0},
			outputs: []float64{v0},
		},
		{
			inputs:  []float64{v0, v1},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v0},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v1},
			outputs: []float64{v0},
		},
	}
	n := neural.NewNetwork(2, 3, 1)
	bp := neural.NewBackpropagation(n)
	//inputs := make([]float64, 2)
	outputs := make([]float64, 1)

	epoch := 0
	for epoch < 1000 {
		worst := 0.0
		for _, sample := range samples {
			bp.Learn(sample.inputs, sample.outputs)

			n.SetInputs(sample.inputs)
			n.Calculate()
			n.GetOutputs(outputs)

			mse := neural.MSE(outputs, sample.outputs)
			if mse > worst {
				worst = mse
			}
			fmt.Printf("%.7f %v, %v, \n", mse, sample.outputs, outputs)
		}
		if worst < epsilon {
			fmt.Println(worst)
			break
		}
		epoch++
	}

	fmt.Println(epoch)
}

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}
