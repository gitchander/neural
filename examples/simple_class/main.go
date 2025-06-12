package main

import (
	"fmt"
	"log"

	gone "github.com/gitchander/neural/goneural"
)

const outN = 4

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

	var (
		//layers = gone.MakeLayers("sigmoid", 4, 8, outN)

		layers = []gone.LayerConfig{
			gone.MakeLayerConfig("sigmoid", 4),
			gone.MakeLayerConfig("sigmoid", 8),
			gone.MakeLayerConfig("softmax", outN),
		}
	)

	p, err := gone.NewNeural(layers)
	if err != nil {
		return err
	}
	p.RandomizeWeights()

	const (
		learnRate = 0.6
		epochMax  = 100000
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

func makeSamples() []gone.Sample {

	// https://www.youtube.com/watch?v=ILsA4nyG7I0

	const (
		solid = iota
		horizontal
		vertical
		diagonal
	)

	samples := []gone.Sample{
		// solid
		{
			Inputs: []float64{
				0, 0,
				0, 0,
			},
			Outputs: gone.OneHot(outN, solid),
		},
		{
			Inputs: []float64{
				1, 1,
				1, 1,
			},
			Outputs: gone.OneHot(outN, solid),
		},
		// horizontal
		{
			Inputs: []float64{
				1, 1,
				0, 0,
			},
			Outputs: gone.OneHot(outN, horizontal),
		},
		{
			Inputs: []float64{
				0, 0,
				1, 1,
			},
			Outputs: gone.OneHot(outN, horizontal),
		},
		// vertical
		{
			Inputs: []float64{
				1, 0,
				1, 0,
			},
			Outputs: gone.OneHot(outN, vertical),
		},
		{
			Inputs: []float64{
				0, 1,
				0, 1,
			},
			Outputs: gone.OneHot(outN, vertical),
		},
		// diagonal
		{
			Inputs: []float64{
				1, 0,
				0, 1,
			},
			Outputs: gone.OneHot(outN, diagonal),
		},
		{
			Inputs: []float64{
				0, 1,
				1, 0,
			},
			Outputs: gone.OneHot(outN, diagonal),
		},
	}
	return samples
}
