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
	//xor()
	xorDraw()
}

func xor() {

	const epsilon = 0.001

	const (
		v0 = 0.1
		v1 = 0.9
		//		v0 = 0.0
		//		v1 = 1.0
	)
	samples := []neural.Sample{
		{
			Inputs:  []float64{v0, v0},
			Outputs: []float64{v0},
		},
		{
			Inputs:  []float64{v0, v1},
			Outputs: []float64{v1},
		},
		{
			Inputs:  []float64{v1, v0},
			Outputs: []float64{v1},
		},
		{
			Inputs:  []float64{v1, v1},
			Outputs: []float64{v0},
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
			bp.Learn(sample)

			p.SetInputs(sample.Inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			mse := neural.MSE(sample.Outputs, outputs)
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
		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g XOR %g = %f\n",
			sample.Inputs[0], sample.Inputs[1],
			outputs[0])
	}

	//p.PrintWeights()
}

func xorDraw() {

	const epsilon = 0.001

	const (
		v0 = 0.1
		v1 = 0.9
		//		v0 = 0.0
		//		v1 = 1.0
	)
	samples := []neural.Sample{
		{
			Inputs:  []float64{v0, v0},
			Outputs: []float64{v0},
		},
		{
			Inputs:  []float64{v0, v1},
			Outputs: []float64{v1},
		},
		{
			Inputs:  []float64{v1, v0},
			Outputs: []float64{v1},
		},
		{
			Inputs:  []float64{v1, v1},
			Outputs: []float64{v0},
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
			bp.Learn(sample)

			p.SetInputs(sample.Inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			mse := neural.MSE(sample.Outputs, outputs)
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
		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		fmt.Printf("%g XOR %g = %f\n",
			sample.Inputs[0], sample.Inputs[1],
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

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}
