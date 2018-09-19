package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
	"github.com/gocarina/gocsv"
)

func main() {
	ps, err := readParamsGo("iris.csv")
	checkError(err)
	samples := makeSamples(ps)
	neutil.NormalizeInputs(samples)
	p, err := neural.NewMLP(4, 3, 3)
	checkError(err)
	p.RandomizeWeights(neutil.NewRand())
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.6)
	const epsilon = 0.001
	epochMax := 1000
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

func readParams(filename string) (ps []*Params, err error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bytes.NewReader(data))

	skipFirst := true
	for {
		record, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if skipFirst {
			skipFirst = false
		} else {
			p, err := parseParams(record)
			checkError(err)
			ps = append(ps, p)
		}
	}
	return ps, nil
}

func readParamsGo(filename string) (ps []*Params, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	err = gocsv.UnmarshalFile(file, &ps)
	return ps, err
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

type Params struct {
	SepalLength float64 `csv:"sepal_length"`
	SepalWidth  float64 `csv:"sepal_width"`
	PetalLength float64 `csv:"petal_length"`
	PetalWidth  float64 `csv:"petal_width"`
	Species     string  `csv:"species"`
}

func parseParams(record []string) (*Params, error) {
	if len(record) != 5 {
		return nil, fmt.Errorf("invalid number of parameters: %d", len(record))
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

func makeSamples(ps []*Params) (samples []neural.Sample) {
	m := make(map[string]int)
	index := 0
	for _, p := range ps {
		if _, ok := m[p.Species]; !ok {
			m[p.Species] = index
			index++
		}
	}
	for _, p := range ps {
		sample := neural.Sample{
			Inputs: []float64{
				p.SepalLength,
				p.SepalWidth,
				p.PetalLength,
				p.PetalWidth,
			},
			Outputs: neutil.OneHot(len(m), m[p.Species]),
		}
		samples = append(samples, sample)
	}
	return samples
}
