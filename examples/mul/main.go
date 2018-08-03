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
}

func mul() {

	const epsilon = 0.01

	p := neural.NewPerceptron(2, 3, 1)
	r := newRand()

	p.RandomizeWeights(r)

	bp := neural.NewBackpropagation()
	bp.SetLearningRate(0.5)

	var sample = neural.Sample{
		Inputs:  make([]float64, 2),
		Outputs: make([]float64, 1),
	}

	samplesInEpoch := 1000

	epoch := 0
	epochMax := 1000
	for epoch < epochMax {

		worst := 0.0
		for i := 0; i < samplesInEpoch; i++ {

			a, b := r.Float64(), r.Float64()
			sample.Inputs[0] = a
			sample.Inputs[1] = b
			sample.Outputs[0] = a * b

			bp.Learn(p, sample)
			mse := p.CalculateMSE(sample)
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

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}
