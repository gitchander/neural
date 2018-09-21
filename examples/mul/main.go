package main

import (
	"fmt"
	"log"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
)

func main() {

	r := neutil.NewRand()
	p, err := neural.NewMLP(2, 3, 1)
	checkError(err)
	p.RandomizeWeightsRand(r)
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.5)

	var samples = make([]neural.Sample, 1000)
	for i := range samples {
		a, b := r.Float64(), r.Float64()
		samples[i] = neural.Sample{
			Inputs:  []float64{a, b},
			Outputs: []float64{a * b},
		}
	}

	const epsilon = 0.001
	epoch := 0
	epochMax := 1000
	for epoch < epochMax {
		averageCost, err := bp.LearnSamples(samples)
		checkError(err)
		if averageCost < epsilon {
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
		log.Fatal(err)
	}
}
