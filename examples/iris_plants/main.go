package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"strconv"
	"time"

	"github.com/gitchander/neural"
)

func main() {
	data, err := ioutil.ReadFile("iris_dataset.csv")
	checkError(err)
	r := csv.NewReader(bytes.NewReader(data))
	var ps []*Params
	for {
		record, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			checkError(err)
		}
		//fmt.Println(record)
		p, err := parseParams(record)
		checkError(err)
		ps = append(ps, p)
	}

	//fmt.Println(len(ps))

	m := make(map[string]int)
	index := 0
	for _, p := range ps {
		if _, ok := m[p.Species]; !ok {
			m[p.Species] = index
			index++
		}
	}

	var samples []neural.Sample
	for _, p := range ps {

		outputs := make([]float64, len(m))
		outputs[m[p.Species]] = 1

		sample := neural.Sample{
			Inputs: []float64{
				p.SepalLength,
				p.SepalWidth,
				p.PetalLength,
				p.PetalWidth,
			},
			Outputs: outputs,
		}
		//fmt.Println(sample.Outputs)
		samples = append(samples, sample)
	}

	p := neural.NewPerceptron(4, 4, 3)

	p.RandomizeWeights(rand.New(rand.NewSource(time.Now().UnixNano())))
	bp := neural.NewBackpropagation()
	bp.SetLearningRate(0.1)
	const epsilon = 0.01

	epochMax := 1000
	for epoch := 0; epoch < epochMax; epoch++ {
		var worst float64
		for _, sample := range samples {
			bp.Learn(p, sample)
			mse := p.CalculateMSE(sample)
			if mse > worst {
				worst = mse
			}
		}
		if worst < epsilon {
			fmt.Println("Success!")
			fmt.Printf("mse: %.7f\n", worst)
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

type Params struct {
	SepalLength float64
	SepalWidth  float64
	PetalLength float64
	PetalWidth  float64
	Species     string
}

func parseParams(record []string) (*Params, error) {
	if len(record) != 5 {
		return nil, fmt.Errorf("invalid number parameters: %d", len(record))
	}
	var vs [4]float64
	for i := range vs {
		v, err := strconv.ParseFloat(record[i], 64)
		if err != nil {
			return nil, err
		}
		vs[i] = v
	}
	return &Params{
		SepalLength: vs[0],
		SepalWidth:  vs[1],
		PetalLength: vs[2],
		PetalWidth:  vs[3],
		Species:     record[4],
	}, nil
}