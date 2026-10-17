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

	samples := makeSamples()

	nnConfig := gone.NetworkConfig{
		Layers: []gone.LayerConfig{
			gone.Dense(4, gone.Sigmoid),
			gone.Dense(20, gone.Sigmoid),
			gone.Dense(7, gone.Sigmoid),
		},
	}
	p, err := gone.NewNeural(nnConfig)
	if err != nil {
		return err
	}
	p.RandomizeWeights()

	const (
		learnRate = 0.7
		epochMax  = 100000
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

	return gone.Train(p, samples, learnRate, epochMax, f)
}
