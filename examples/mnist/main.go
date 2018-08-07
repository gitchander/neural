package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"path/filepath"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/dataset/mnist"
)

func main() {

	var dirname string
	flag.StringVar(&dirname, "dir", ".", "mnist directory")

	flag.Parse()

	var (
		nameImages = filepath.Join(dirname, "train-images-idx3-ubyte.gz")
		nameLabels = filepath.Join(dirname, "train-labels-idx1-ubyte.gz")
	)

	samples, err := makeSamples(nameImages, nameLabels)
	checkError(err)

	fmt.Println(len(samples))

	p := neural.NewPerceptron(28*28, 1000, 10)
	p.RandomizeWeights(neural.NewRand())
	bp := neural.NewBackpropagation(p)
	bp.SetLearningRate(0.9)

	//outputs := make([]float64, 10)
	//	epochMax := 1000
	//	for epoch := 0; epoch < epochMax; epoch++ {
	//		for i, sample := range samples[:10] {
	//			err := bp.Learn(sample)
	//			checkError(err)
	//			//			mse := p.CalculateMSE(sample)
	//			//			fmt.Printf("%d: mse = %.15f\n", i, mse)
	//			p.SetInputs(sample.Inputs)
	//			p.Calculate()
	//			p.GetOutputs(outputs)
	//			fmt.Printf("%d:\n", i)
	//			fmt.Println(sample.Outputs)
	//			for _, v := range outputs {
	//				fmt.Printf("%.5f ", v)
	//			}
	//			fmt.Println()
	//			fmt.Printf("mse = %.15f\n", neural.MSE(sample.Outputs, outputs))
	//			fmt.Println()
	//		}
	//	}
	//----------------------------------------
	epochMax := 100000
	for epoch := 0; epoch < epochMax; epoch++ {
		mse, err := bp.LearnSamples(samples[:100])
		checkError(err)
		fmt.Printf("%d: mse = %.15f\n", epoch, mse)
	}
	//----------------------------------------
	//	subSamples := samples[:20]
	//	epochMax := 100000
	//	for epoch := 0; epoch < epochMax; epoch++ {
	//		var sumMSE float64
	//		for _, sample := range subSamples {
	//			bp.Learn(sample)
	//			mse := p.CalculateMSE(sample)
	//			sumMSE += mse
	//		}
	//		fmt.Printf("%d: mse = %.15f\n", epoch, sumMSE/float64(len(subSamples)))
	//	}
}

func makeSamples(nameImages, nameLabels string) ([]neural.Sample, error) {
	images, err := mnist.ReadImagesFile(nameImages)
	if err != nil {
		return nil, err
	}
	labels, err := mnist.ReadLabelsFile(nameLabels)
	if err != nil {
		return nil, err
	}

	n := len(images)
	if n != len(labels) {
		return nil, errors.New("number of images not equal number of labels")
	}

	samples := make([]neural.Sample, n)

	for i, g := range images {
		inputs := make([]float64, len(g.Pix))
		for j, p := range g.Pix {
			inputs[j] = float64(p) / 255
		}
		outputs := make([]float64, 10)
		for i := range outputs {
			outputs[i] = 0
		}
		outputs[labels[i]] = 1

		samples[i] = neural.Sample{
			Inputs:  inputs,
			Outputs: outputs,
		}
	}

	return samples, nil
}

func checkError(err error) {
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
