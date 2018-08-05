package main

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/gitchander/neural"
)

func main() {

	dir := "../../../../../neural_samples/mnist/samples"

	var (
		nameImages = filepath.Join(dir, "train-images.idx3-ubyte")
		nameLabels = filepath.Join(dir, "train-labels.idx1-ubyte")
	)

	p := neural.NewPerceptron(28*28, 1500, 10)
	p.RandomizeWeights(rand.New(rand.NewSource(time.Now().UnixNano())))
	bp := neural.NewBackpropagation()
	bp.SetLearningRate(0.7)

	var (
		inputs  = make([]float64, 28*28)
		outputs = make([]float64, 10)
	)

	i := 0
	f := func(size image.Point, data []byte, label byte) bool {

		//		if i == 1001 {
		//			g, err := imageFromData(size, data)
		//			if err != nil {
		//				log.Fatal(err)
		//			}
		//			err = saveImagePNG(g, "test.png")
		//			if err != nil {
		//				log.Fatal(err)
		//			}
		//			fmt.Println(label)
		//			return false
		//		}

		//		if i > 10 {
		//			return false
		//		}
		for i := range inputs {
			inputs[i] = float64(data[i]) / 255
		}
		for i := range outputs {
			outputs[i] = 0
		}
		outputs[label] = 1

		sample := neural.Sample{
			Inputs:  inputs,
			Outputs: outputs,
		}
		bp.Learn(p, sample)
		mse := p.CalculateMSE(sample)
		fmt.Printf("%d, mse = %.8f\n", i, mse)

		i++
		return true
	}

	err := WalkMNIST(nameImages, nameLabels, f)
	if err != nil {
		log.Fatal(err)
	}
}

func saveImagePNG(im image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, im)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0664)
}
