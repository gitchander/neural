package main

import (
	"fmt"
	"log"

	gone "github.com/gitchander/neural/goneural"
	"github.com/gitchander/neural/neutil/random"
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
		// layers = gone.MakeLayers("sigmoid", 4, 3, 3)

		layers = []gone.LayerConfig{
			gone.MakeLayerConfig("sigmoid", 4),
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
		learnRate = 0.1
		epochMax  = 100_000
		epsilon   = 0.01
	)

	r := random.NewRandNow()
	shuffleSamples := func() {
		r.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
	}
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

	return gone.Learn(p, samples, learnRate, epochMax, f)
}
