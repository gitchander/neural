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
	testOperator(XOR)
	//makeOperatorImage(AND)
}

func testOperator(op operator) {

	samples := makeSamplesByOperator(op)

	p := neural.NewPerceptron(2, 3, 1)
	p.RandomizeWeights(neural.NewRand())
	bp := neural.NewBackpropagation(p)
	bp.SetLearningRate(0.6)

	outputs := make([]float64, 1)

	epoch := 0
	epochMax := 10000
	const epsilon = 0.001
	for epoch < epochMax {
		mse, err := bp.LearnSamples(samples)
		checkError(err)
		if mse < epsilon {
			fmt.Printf("mse: %.7f\n", mse)
			break
		}
		epoch++
	}

	if epoch == epochMax {
		fmt.Println("failure")
		return
	}

	fmt.Println("epoch:", epoch)

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

	p := neural.NewPerceptron(2, 3, 1)
	p.RandomizeWeights(neural.NewRand())
	bp := neural.NewBackpropagation(p)
	bp.SetLearningRate(0.5)

	epoch := 0
	epochMax := 1000
	const epsilon = 0.01
	for epoch < epochMax {
		worst := 0.0
		for _, sample := range samples {
			bp.Learn(sample)
			mse := p.CalculateMSE(sample)
			if mse > worst {
				worst = mse
			}
		}
		if worst < epsilon {
			fmt.Printf("mse: %.7f\n", worst)
			break
		}
		epoch++
	}

	if epoch == epochMax {
		fmt.Println("failure")
		return
	}

	fmt.Println("epoch:", epoch)

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
	err := png.Encode(&buf, m)
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
		return 0.9
	}
	return 0.1
}
