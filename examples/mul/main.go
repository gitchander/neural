package main

import (
	"fmt"
	"log"
	"math/rand"

	gone "github.com/gitchander/neural/goneural"
	"github.com/gitchander/neural/neutil/random"
)

func main() {

	r := random.NewRandNow()
	samples := makeSamples(r)

	layers := gone.MakeLayers("sigmoid", 2, 3, 1)
	p, err := gone.NewNeural(layers)
	checkError(err)
	p.RandomizeWeights()

	const (
		learnRate = 0.7
		epochMax  = 1000
		epsilon   = 0.0001
	)

	f := func(epoch int, averageCost float64) bool {
		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		return true
	}

	err = gone.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func makeSamples(r *rand.Rand) []gone.Sample {
	var samples = make([]gone.Sample, 1000)
	for i := range samples {
		a, b := r.Float64(), r.Float64()
		samples[i] = gone.Sample{
			Inputs:  []float64{a, b},
			Outputs: []float64{a * b},
		}
	}
	return samples
}
