package main

import (
	"fmt"
	"log"

	gone "github.com/gitchander/neural/goneural"
)

func main() {
	err := run()
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func run() error {

	samples, err := makeSamplesFile("iris.csv")
	if err != nil {
		return err
	}
	gone.NormalizeInputs(samples)

	var (
		// layers = []gone.LayerConfig{
		// 	gone.MakeLayerConfigAF(4, gone.Sigmoid),
		// 	gone.MakeLayerConfigAF(3, gone.Sigmoid),
		// 	gone.MakeLayerConfigAF(3, gone.Sigmoid),
		// }

		layers = []gone.LayerConfig{
			gone.MakeLayerConfigAF(4, gone.Sigmoid),
			gone.MakeLayerConfigAF(3, gone.Sigmoid),
			gone.MakeLayerConfigSoftmax(3),
		}
	)

	p, err := gone.NewNeural(layers)
	if err != nil {
		return err
	}
	p.RandomizeWeights()

	const (
		learnRate = 0.1
		epochMax  = 1000
		epsilon   = 0.1
	)

	shuffleSamples := gone.ShuffleSamplesFunc(samples)
	shuffleSamples()

	f := func(epoch int, averageCost float64) bool {
		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		shuffleSamples()
		return true
	}

	return gone.Train(p, samples, learnRate, epochMax, f)
}
