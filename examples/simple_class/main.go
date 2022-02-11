package main

import (
	"fmt"
	"log"

	"github.com/gitchander/neural"
)

func main() {

	samples := makeSamples()

	layers := neural.MakeLayers(neural.ActSigmoid, 4, 8, outN)
	p, err := neural.NewNeural(layers)
	checkError(err)
	p.RandomizeWeights()

	const (
		learnRate = 0.6
		epochMax  = 10000
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

const outN = 4

func makeSamples() []neural.Sample {

	// https://www.youtube.com/watch?v=ILsA4nyG7I0

	const (
		solid = iota
		horizontal
		vertical
		diagonal
	)

	samples := []neural.Sample{
		// solid
		{
			Inputs: []float64{
				0, 0,
				0, 0,
			},
			Outputs: neural.OneHot(outN, solid),
		},
		{
			Inputs: []float64{
				1, 1,
				1, 1,
			},
			Outputs: neural.OneHot(outN, solid),
		},
		// horizontal
		{
			Inputs: []float64{
				1, 1,
				0, 0,
			},
			Outputs: neural.OneHot(outN, horizontal),
		},
		{
			Inputs: []float64{
				0, 0,
				1, 1,
			},
			Outputs: neural.OneHot(outN, horizontal),
		},
		// vertical
		{
			Inputs: []float64{
				1, 0,
				1, 0,
			},
			Outputs: neural.OneHot(outN, vertical),
		},
		{
			Inputs: []float64{
				0, 1,
				0, 1,
			},
			Outputs: neural.OneHot(outN, vertical),
		},
		// diagonal
		{
			Inputs: []float64{
				1, 0,
				0, 1,
			},
			Outputs: neural.OneHot(outN, diagonal),
		},
		{
			Inputs: []float64{
				0, 1,
				1, 0,
			},
			Outputs: neural.OneHot(outN, diagonal),
		},
	}
	return samples
}
