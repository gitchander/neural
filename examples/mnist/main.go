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
	"github.com/gitchander/neural/neutil"
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

func train(dirname, nameMLP string) {

	var (
		nameImages = filepath.Join(dirname, "train-images-idx3-ubyte.gz")
		nameLabels = filepath.Join(dirname, "train-labels-idx1-ubyte.gz")
	)

	samples, err := mnist.MakeSamples(nameImages, nameLabels)
	checkError(err)

	//fmt.Println(len(samples))

	p, err := neural.ReadFile(nameMLP)
	if err != nil {
		p, err = neural.NewMLP(28*28, 800, 10)
		checkError(err)
		p.RandomizeWeights(neutil.NewRand())
	}

	bp := neural.NewBP(p)
	bp.SetLearningRate(0.6)

	epochMax := 100000
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost, err := bp.LearnSamples(samples)
		checkError(err)

		err = neural.WriteFile(nameMLP, p)
		checkError(err)

		fmt.Printf("%s: epoch = %d; average cost = %.10f\n",
			time.Now().Format("15:04:05"),
			epoch, averageCost)
	}
}

func test(dirname, nameMLP string) {
	var (
		nameImages = filepath.Join(dirname, "t10k-images-idx3-ubyte.gz")
		nameLabels = filepath.Join(dirname, "t10k-labels-idx1-ubyte.gz")
	)

	images, err := mnist.ReadImagesFile(nameImages)
	checkError(err)

	labels, err := mnist.ReadLabelsFile(nameLabels)
	checkError(err)

	p, err := neural.ReadFile(nameMLP)
	checkError(err)

	outputs := make([]float64, 10)

	var wrongCount int
	for i, g := range images {
		inputs := mnist.InputsFromImage(g)

		err = p.SetInputs(inputs)
		checkError(err)

		p.Calculate()

		err = p.GetOutputs(outputs)
		checkError(err)

		var (
			labelIdeal = int(labels[i])
			label      = maxFloat64Index(outputs)
		)
		if labelIdeal != label {
			//			name := filepath.Join("bad_images", fmt.Sprintf("test_%05d_%d_%d.png", i, labelIdeal, label))
			//			err = saveImagePNG(g, name)
			//			checkError(err)

			//fmt.Printf("%d: (%d != %d)\n", i, labelIdeal, label)

			wrongCount++
		}
	}
	fmt.Printf("error rate: %.3f %%\n", 100*float64(wrongCount)/float64(len(images)))
}

func maxFloat64Index(vs []float64) (max int) {
	n := len(vs)
	if n == 0 {
		return -1
	}
	for i := 1; i < n; i++ {
		if vs[i] > vs[max] {
			max = i
		}
	}
	return max
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
