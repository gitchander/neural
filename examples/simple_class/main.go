package main

import (
	"fmt"
	"log"

	"github.com/gitchander/neural"
)

func main() {

	// https://www.youtube.com/watch?v=ILsA4nyG7I0

	const outN = 4

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
	p, err := neural.NewMLP(4, 8, outN)
	checkError(err)
	p.RandomizeWeights()
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.6)
	const epsilon = 0.001
	epochMax := 10000
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost, err := bp.LearnSamples(samples)
		checkError(err)
		if averageCost < epsilon {
			fmt.Println("Success!")
			fmt.Printf("average cost: %.7f\n", averageCost)
			fmt.Println("epoch =", epoch)
			return
		}
	}
	fmt.Println("Failure")
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
