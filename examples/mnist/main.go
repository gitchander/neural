package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"time"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/dataset/mnist"
	"github.com/gitchander/neural/neutil"
)

func main() {

	var c Config

	flag.StringVar(&(c.Dirname), "dir", ".", "mnist directory")
	flag.StringVar(&(c.Mode), "mode", "test", "train or test mode")
	flag.StringVar(&(c.NeuralName), "neural", "neural", "filename neural network")

	flag.Parse()

	err := run(c)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

type Config struct {
	Dirname    string
	Mode       string
	NeuralName string
}

func run(c Config) error {
	switch c.Mode {
	case "train":
		return train(c.Dirname, c.NeuralName)
	case "test":
		return test(c.Dirname, c.NeuralName)
	default:
		return fmt.Errorf("invalid mode: %s", c.Mode)
	}
}

func train(dirname, neuralName string) error {

	dbfs := mnist.MakeDBFiles(dirname)

	samples, err := mnist.MakeSamples(dbfs.TrainingSet)
	if err != nil {
		return err
	}

	p, err := neural.ReadFile(neuralName)
	if err != nil {
		layers := neural.MakeLayers(neural.ActSigmoid, 28*28, 14*14, 7*7, 10)
		p, err = neural.NewNeural(layers)
		if err != nil {
			return err
		}
		p.RandomizeWeights()
	}

	const (
		learnRate = 0.6
		epochMax  = 100000
	)

	var writeError error

	f := func(epoch int, averageCost float64) bool {

		err := neural.WriteFile(neuralName, p)
		if err != nil {
			writeError = err
			return false
		}

		fmt.Printf("%s: epoch: %d; average cost = %.10f\n",
			time.Now().Format("15:04:05"),
			epoch, averageCost)

		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	if err != nil {
		return err
	}

	return writeError
}

func test(dirname, neuralName string) error {

	dbfs := mnist.MakeDBFiles(dirname)

	inputs, err := mnist.ReadInputsFile(dbfs.TestSet.Images)
	if err != nil {
		return err
	}

	labels, err := mnist.ReadLabelsFile(dbfs.TestSet.Labels)
	if err != nil {
		return err
	}

	p, err := neural.ReadFile(neuralName)
	if err != nil {
		return err
	}

	fmt.Println("topology:", p.Topology())

	outputs := make([]float64, 10)

	var wrongCount int
	for i := range inputs {
		p.SetInputs(inputs[i])
		p.Calculate()
		p.GetOutputs(outputs)

		var (
			labelIdeal = int(labels[i])
			label      = neutil.IndexOfMax(outputs)
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

	return nil
}

func saveImagePNG(im image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, im)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0664)
}
