package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"

	"github.com/gitchander/neural"
)

func main() {
	//testOperator(XOR)
	makeOperatorImage(XOR)
	//testNot()
}

func testOperator(op operator) {

	samples := makeSamplesByOperator(op)

	layers := neural.MakeLayers(neural.ActSigmoid, 2, 3, 1)
	p, err := neural.NewNeural(layers)
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

	outputs := make([]float64, 1)
	for _, sample := range samples {
		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g %s %g = %f\n",
			sample.Inputs[0],
			op.name,
			sample.Inputs[1],
			outputs[0])
	}
}

func makeOperatorImage(op operator) {

	samples := makeSamplesByOperator(op)

	layers := neural.MakeLayers(neural.ActSigmoid, 2, 3, 1)
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

	var (
		inputs  = make([]float64, 2)
		outputs = make([]float64, 1)
	)

	for _, sample := range samples {
		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g %s %g = %f\n",
			sample.Inputs[0],
			op.name,
			sample.Inputs[1],
			outputs[0])
	}

	size := image.Pt(512, 512)
	m := image.NewGray(image.Rect(0, 0, size.X, size.Y))
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {

			inputs[0] = float64(x) / float64(size.X)
			inputs[1] = float64(y) / float64(size.Y)

			p.SetInputs(inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			gray := color.Gray{
				Y: uint8(outputs[0] * 256),
			}

			m.SetGray(x, y, gray)
		}
	}

	var buf bytes.Buffer
	err = png.Encode(&buf, m)
	checkError(err)
	filename := fmt.Sprintf("op_%s.png", op.name)
	err = ioutil.WriteFile(filename, buf.Bytes(), 0666)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

type operator struct {
	name string
	f    func(a, b bool) bool
}

var (
	OR = operator{
		name: "OR",
		f:    func(a, b bool) bool { return a || b },
	}

	AND = operator{
		name: "AND",
		f:    func(a, b bool) bool { return a && b },
	}

	XOR = operator{
		name: "XOR",
		f:    func(a, b bool) bool { return a != b },
	}
)

func makeSamplesByOperator(op operator) (samples []neural.Sample) {
	var bs = []bool{false, true}
	for _, b1 := range bs {
		for _, b2 := range bs {
			sample := neural.Sample{
				Inputs: []float64{
					boolToFloat(b1),
					boolToFloat(b2),
				},
				Outputs: []float64{
					boolToFloat(op.f(b1, b2)),
				},
			}
			samples = append(samples, sample)
		}
	}
	return samples
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func testNot() {
	samples := []neural.Sample{
		{
			Inputs:  []float64{0},
			Outputs: []float64{1},
		},
		{
			Inputs:  []float64{1},
			Outputs: []float64{0},
		},
	}
	layers := neural.MakeLayers(neural.ActSigmoid, 1, 1)
	p, err := neural.NewNeural(layers)
	checkError(err)
	p.RandomizeWeights()

	const (
		learnRate = 0.8
		epochMax  = 1000
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
