package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"time"

	"github.com/gitchander/neural"
)

func main() {
	//mul()
	xor()
	//xorDraw()
}

func test() {
	p := neural.NewPerceptron(2, 3, 1)
	p.RandomizeWeights(newRand())

	inputs := []float64{0.7, 0.0}

	err := p.SetInputs(inputs)
	checkError(err)

	p.Calculate()

	outputs := make([]float64, 1)
	p.GetOutputs(outputs)
	fmt.Println(outputs)

	bp := neural.NewBackpropagation(p)

	for i := 0; i < 1000; i++ {
		bp.Learn(inputs, []float64{0.5})
	}

	err = p.SetInputs(inputs)
	checkError(err)

	p.Calculate()
	p.GetOutputs(outputs)
	fmt.Println(outputs)
}

func mul() {

	const epsilon = 0.001

	p := neural.NewPerceptron(2, 3, 2, 1)
	r := newRand()
	p.RandomizeWeights(r)
	bp := neural.NewBackpropagation(p)

	var (
		inputs       = make([]float64, 2)
		outputs      = make([]float64, 1)
		outputsIdeal = make([]float64, 1)
	)

	samplesInEpoch := 1000

	epoch := 0
	epochMax := 10000
	for epoch < epochMax {

		worst := 0.0
		for i := 0; i < samplesInEpoch; i++ {

			a, b := r.Float64(), r.Float64()
			inputs[0] = a
			inputs[1] = b
			outputsIdeal[0] = a * b

			bp.Learn(inputs, outputsIdeal)

			p.SetInputs(inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			mse := neural.MSE(outputs, outputsIdeal)
			if mse > worst {
				worst = mse
			}
		}

		if worst < epsilon {
			break
		}

		epoch++
	}
	if epoch < epochMax {
		fmt.Println("success: epoch =", epoch)
	} else {
		fmt.Println("failure")
	}
}

type sample struct {
	inputs  []float64
	outputs []float64
}

func xor() {

	const epsilon = 0.001

	const (
		v0 = 0.1
		v1 = 0.9
		//		v0 = 0.0
		//		v1 = 1.0
	)
	samples := []sample{
		{
			inputs:  []float64{v0, v0},
			outputs: []float64{v0},
		},
		{
			inputs:  []float64{v0, v1},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v0},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v1},
			outputs: []float64{v0},
		},
	}
	p := neural.NewPerceptron(2, 3, 1)
	p.RandomizeWeights(newRand())
	bp := neural.NewBackpropagation(p)

	outputs := make([]float64, 1)

	epoch := 0
	epochMax := 10000
	for epoch < epochMax {
		worst := 0.0
		for _, sample := range samples {
			bp.Learn(sample.inputs, sample.outputs)

			p.SetInputs(sample.inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			mse := neural.MSE(sample.outputs, outputs)
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

	for _, sample := range samples {
		p.SetInputs(sample.inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g XOR %g = %f\n",
			sample.inputs[0], sample.inputs[1],
			outputs[0])
	}
}

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func xorDraw() {

	const epsilon = 0.001

	const (
		v0 = 0.1
		v1 = 0.9
		//		v0 = 0.0
		//		v1 = 1.0
	)
	samples := []sample{
		{
			inputs:  []float64{v0, v0},
			outputs: []float64{v0},
		},
		{
			inputs:  []float64{v0, v1},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v0},
			outputs: []float64{v1},
		},
		{
			inputs:  []float64{v1, v1},
			outputs: []float64{v0},
		},
	}
	p := neural.NewPerceptron(2, 3, 1)
	p.RandomizeWeights(newRand())
	bp := neural.NewBackpropagation(p)

	inputs := make([]float64, 2)
	outputs := make([]float64, 1)

	epoch := 0
	epochMax := 10000
	for epoch < epochMax {
		worst := 0.0
		for _, sample := range samples {
			bp.Learn(sample.inputs, sample.outputs)

			p.SetInputs(sample.inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			mse := neural.MSE(sample.outputs, outputs)
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

	for _, sample := range samples {
		p.SetInputs(sample.inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g XOR %g = %f\n",
			sample.inputs[0], sample.inputs[1],
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
	err = ioutil.WriteFile("xor.png", buf.Bytes(), 0666)
	checkError(err)
}
