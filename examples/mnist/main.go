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
	"github.com/gitchander/neural/neutil"
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

	p, err := neural.NewMLP(28*28, 14*14, 7*7, 10)
	checkError(err)
	p.RandomizeWeights(neutil.NewRand())
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.6)

	encodeNeural(p)

	//----------------------------------------
	//	epochMax := 100000
	//	for epoch := 0; epoch < epochMax; epoch++ {
	//		le, err := bp.LearnSamples(samples[:1000])
	//		checkError(err)
	//		fmt.Printf("%d: error = %.15f\n", epoch, le)
	//	}
	//----------------------------------------
	subSamples := samples[:]
	i := 0
	var st neutil.Statistics
	epochMax := 100000
	for epoch := 0; epoch < epochMax; epoch++ {
		for _, sample := range subSamples {
			bp.Learn(sample)
			ce := p.SampleError(sample)
			st.Add(ce)
			i++
			if (i % 1000) == 0 {
				fmt.Printf("epoch=%d; error=%.10f\n", epoch, st.Mean())
				st.Reset()
			}
		}
	}
}

func encodeNeural(p *neural.MLP) {
	var buf bytes.Buffer
	err := neural.Encode(&buf, p)
	checkError(err)
	fmt.Println("len =", len(buf.Bytes()))
	q, err := neural.Decode(&buf)
	checkError(err)
	fmt.Println(neural.Equal(p, q))
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
