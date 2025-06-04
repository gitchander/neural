package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"

	gone "github.com/gitchander/neural/goneural"
	"github.com/gocarina/gocsv"
)

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
			if err != nil {
				return nil, err
			}
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

func makeSamples(ps []*Params) (samples []gone.Sample) {
	m := make(map[string]int)
	index := 0
	for _, p := range ps {
		if _, ok := m[p.Species]; !ok {
			m[p.Species] = index
			index++
		}
	}
	for _, p := range ps {
		sample := gone.Sample{
			Inputs: []float64{
				p.SepalLength,
				p.SepalWidth,
				p.PetalLength,
				p.PetalWidth,
			},
			Outputs: gone.OneHot(len(m), m[p.Species]),
		}
		samples = append(samples, sample)
	}
	return samples
}

func makeSamplesFile(filename string) (samples []gone.Sample, err error) {
	ps, err := readParamsGo(filename)
	if err != nil {
		return nil, err
	}
	samples = makeSamples(ps)
	return samples, nil
}
