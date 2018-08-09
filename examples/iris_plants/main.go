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
	"github.com/gocarina/gocsv"
)

func main() {

	ps, err := readParams2("iris_dataset.csv")
	checkError(err)
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

	p := neural.NewPerceptron(4, 3, 3)
	p.RandomizeWeights(neural.NewRand())
	bp := neural.NewBackpropagation(p)
	bp.SetLearningRate(0.6)
	const epsilon = 0.001
	epochMax := 1000
	for epoch := 0; epoch < epochMax; epoch++ {
		le, err := bp.LearnSamples(samples)
		checkError(err)
		if le < epsilon {
			fmt.Println("Success!")
			fmt.Printf("error: %.7f\n", le)
			fmt.Println("epoch =", epoch)
			return
		}
	}
	fmt.Println("Failure")
}

func readParams1(filename string) (ps []*Params, err error) {
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

func readParams2(filename string) (ps []*Params, err error) {
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
