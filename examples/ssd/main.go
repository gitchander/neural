package main

import (
	"fmt"
	"log"

	"github.com/gitchander/neural"
)

func main() {

	samples := makeSamples()

	p, err := neural.NewMLP(4, 20, 7)
	checkError(err)
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

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
