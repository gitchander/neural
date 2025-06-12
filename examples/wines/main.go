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

	var (
		// layers = gone.MakeLayers("sigmoid", 13, 3, 3)

		layers = []gone.LayerConfig{
			gone.MakeLayerConfig("sigmoid", 13),
			gone.MakeLayerConfig("sigmoid", 3),
			gone.MakeLayerConfig("softmax", 3),
		}
	)

	p, err := gone.NewNeural(layers)
	if err != nil {
		return err
	}

	p.RandomizeWeights()

	const (
		learnRate = 0.6
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

	return gone.Learn(p, samples, learnRate, epochMax, f)
}
