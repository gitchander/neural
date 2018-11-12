package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"path/filepath"
	"time"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/dataset/mnist"
)

func main() {

	var (
		dirname string
		mode    string
		nameMLP string
	)
	flag.StringVar(&dirname, "dir", ".", "mnist directory")
	flag.StringVar(&mode, "mode", "test", "train or test mode")
	flag.StringVar(&nameMLP, "mlp", "mlp1ws", "filename MLP structure")

	flag.Parse()

	switch mode {
	case "train":
		train(dirname, nameMLP)
	case "test":
		test(dirname, nameMLP)
	default:
		log.Fatalf("invalid mode: %s", mode)
	}
}

type names struct {
	imagesName string
	labelsName string
}

var (
	trainNames = names{
		imagesName: "train-images-idx3-ubyte.gz",
		labelsName: "train-labels-idx1-ubyte.gz",
	}

	testNames = names{
		imagesName: "t10k-images-idx3-ubyte.gz",
		labelsName: "t10k-labels-idx1-ubyte.gz",
	}
)

func train(dirname, nameMLP string) {

	var (
		ns = trainNames

		nameImages = filepath.Join(dirname, ns.imagesName)
		nameLabels = filepath.Join(dirname, ns.labelsName)
	)

	samples, err := mnist.MakeSamples(nameImages, nameLabels)
	checkError(err)

	p, err := neural.ReadFile(nameMLP)
	if err != nil {
		p, err = neural.NewMLP(28*28, 14*14, 7*7, 10)
		checkError(err)
		p.RandomizeWeights()
	}

	const (
		learnRate = 0.6
		epochMax  = 100000
	)

	f := func(epoch int, averageCost float64) bool {

		err = neural.WriteFile(nameMLP, p)
		checkError(err)

		fmt.Printf("%s: epoch: %d; average cost = %.10f\n",
			time.Now().Format("15:04:05"),
			epoch, averageCost)

		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)
}

func test(dirname, nameMLP string) {
	var (
		ns = testNames
		//ns = trainNames

		nameImages = filepath.Join(dirname, ns.imagesName)
		nameLabels = filepath.Join(dirname, ns.labelsName)
	)

	inputs, err := mnist.ReadInputsFile(nameImages)
	checkError(err)

	labels, err := mnist.ReadLabelsFile(nameLabels)
	checkError(err)

	p, err := neural.ReadFile(nameMLP)
	checkError(err)

	fmt.Println("topology:", p.Topology())

	outputs := make([]float64, 10)

	var wrongCount int
	for i := range inputs {
		p.SetInputs(inputs[i])
		p.Calculate()
		p.GetOutputs(outputs)

		var (
			labelIdeal = int(labels[i])
			label      = neural.IndexOfMax(outputs)
		)
		if labelIdeal != label {
			//			name := filepath.Join("bad_images", fmt.Sprintf("test_%05d_%d_%d.png", i, labelIdeal, label))
			//			err = saveImagePNG(g, name)
			//			checkError(err)

			//fmt.Printf("%d: (%d != %d)\n", i, labelIdeal, label)

			wrongCount++
		}
	}
	fmt.Printf("average cost: %.3f %%\n", 100*float64(wrongCount)/float64(len(inputs)))
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
