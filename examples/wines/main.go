package main

import (
	"fmt"
	"log"

	gone "github.com/gitchander/neural/goneural"
)

func main() {
	checkError(run())
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func run() error {
	samples, err := makeSamplesFile("wine.csv")
	if err != nil {
		return err
	}
	gone.NormalizeInputs(samples)

	// nnConfig := gone.NetworkConfig{
	// 	Layers: []gone.LayerConfig{
	// 		gone.Dense(13, gone.Sigmoid),
	// 		gone.Dense(3, gone.Sigmoid),
	// 		gone.Dense(3, gone.Sigmoid),
	// 	},
	// }

	nnConfig := gone.NetworkConfig{
		Layers: []gone.LayerConfig{
			gone.Dense(13, gone.Sigmoid),
			gone.Dense(3, gone.Sigmoid),
			gone.Softmax(3),
		},
	}

	p, err := gone.NewNeural(nnConfig)
	if err != nil {
		return err
	}

	p.RandomizeWeights()

	const (
		learnRate = 0.1
		epochMax  = 10000
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

	return gone.Train(p, samples, learnRate, epochMax, f)
}
