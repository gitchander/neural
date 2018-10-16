package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
)

func main() {

	r := neutil.NewRand()
	samples := makeSamples(r)

	p, err := neural.NewMLP(2, 3, 1)
	checkError(err)
	p.RandomizeWeightsRand(r)

	const (
		learnRate = 0.7
		epochMax  = 1000
		epsilon   = 0.001
	)

	f := func(epoch int, averageCost float64) bool {
		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func makeSamples(r *rand.Rand) []neural.Sample {
	var samples = make([]neural.Sample, 1000)
	for i := range samples {
		a, b := r.Float64(), r.Float64()
		samples[i] = neural.Sample{
			Inputs:  []float64{a, b},
			Outputs: []float64{a * b},
		}
	}
	return samples
}
