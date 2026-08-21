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

	"github.com/gitchander/neural/dataset/mnist"
	"github.com/gitchander/neural/dataset/mnist_png"
	gone "github.com/gitchander/neural/goneural"
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
		return test2(c.Dirname, c.NeuralName)
	default:
		return fmt.Errorf("invalid mode: %s", c.Mode)
	}
}

func train(dirname, neuralName string) error {

	samples, err := dsr.ReadTraining(dirname)
	if err != nil {
		return err
	}
	// for _, s := range samples {
	// 	fmt.Println(s.Outputs)
	// 	time.Sleep(time.Millisecond)
	// }
	// return nil

	p, err := gone.ReadFile(neuralName)
	if err != nil {
		var (
			// layers = gone.MakeLayers("sigmoid", 28*28, 14*14, 7*7, 10)

			// layers = []gone.LayerConfig{
			// 	gone.MakeLayerConfig("sigmoid", 28*28),
			// 	gone.MakeLayerConfig("sigmoid", 14*14),
			// 	gone.MakeLayerConfig("sigmoid", 7*7),
			// 	gone.MakeLayerConfig("softmax", 10),
			// }

			// layers = []gone.LayerConfig{
			// 	gone.MakeLayerConfig("sigmoid", 28*28),
			// 	gone.MakeLayerConfig("sigmoid", 800),
			// 	gone.MakeLayerConfig("sigmoid", 10),
			// }

			layers = []gone.LayerConfig{
				gone.MakeLayerConfig("sigmoid", (28 * 28)),
				gone.MakeLayerConfig("sigmoid", 800),
				gone.MakeLayerConfig("softmax", 10),
			}
		)
		p, err = gone.NewNeural(layers)
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

		err := gone.WriteFile(neuralName, p)
		if err != nil {
			writeError = err
			return false
		}

		fmt.Printf("%s: epoch: %d; average cost = %.10f\n",
			time.Now().Format("15:04:05"),
			epoch, averageCost)

		return true
	}

	err = gone.Learn(p, samples, learnRate, epochMax, f)
	if err != nil {
		return err
	}

	return writeError
}

func test1(dirname, neuralName string) error {

	//samples, err := dsr.ReadTraining(dirname)
	samples, err := dsr.ReadTesting(dirname)
	if err != nil {
		return err
	}

	p, err := gone.ReadFile(neuralName)
	if err != nil {
		return err
	}

	fmt.Println("topology:", p.Topology())

	outputs := make([]float64, 10)

	var wrongCount int
	for _, sample := range samples {

		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		//fmt.Println("sum Outputs:", sample.Outputs)
		//fmt.Println(outputs, sample.Outputs)

		var (
			labelIdeal = neutil.IndexOfMax(sample.Outputs)
			label      = neutil.IndexOfMax(outputs)
		)

		//fmt.Println(labelIdeal, label)

		if labelIdeal != label {
			wrongCount++
		}
	}
	fmt.Printf("average cost: %.3f %%\n", 100*float64(wrongCount)/float64(len(samples)))

	return nil
}

func test2(dirname, neuralName string) error {

	//samples, err := dsr.ReadTraining(dirname)
	samples, err := dsr.ReadTesting(dirname)
	if err != nil {
		return err
	}

	p, err := gone.ReadFile(neuralName)
	if err != nil {
		return err
	}

	fmt.Println("topology:", p.Topology())

	outputs := make([]float64, 10)

	cf := gone.CrossEntropy{}
	var sum float64
	for _, sample := range samples {

		p.SetInputs(sample.Inputs)
		p.Calculate()
		p.GetOutputs(outputs)

		//fmt.Println("sum Outputs:", sample.Outputs)
		//fmt.Println(outputs, sample.Outputs)

		sum += cf.Func(sample.Outputs, outputs)
	}
	averageCost := sum / float64(len(samples))

	fmt.Printf("average cost = %.10f\n", averageCost)

	return nil
}

func sumFloat64s(as []float64) float64 {
	sum := 0.0
	for _, a := range as {
		sum += a
	}
	return sum
}

func saveImagePNG(im image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, im)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0664)
}

//------------------------------------------------------------------------------

type DataSetReader interface {
	ReadTraining(dirname string) ([]gone.Sample, error)
	ReadTesting(dirname string) ([]gone.Sample, error)
}

type mnistDSR struct{}

func (mnistDSR) ReadTraining(dirname string) ([]gone.Sample, error) {
	return mnist.ReadTraining(dirname)
}

func (mnistDSR) ReadTesting(dirname string) ([]gone.Sample, error) {
	return mnist.ReadTesting(dirname)
}

type mnistPNG_DSR struct{}

func (mnistPNG_DSR) ReadTraining(dirname string) ([]gone.Sample, error) {
	return mnist_png.ReadTraining(dirname)
}

func (mnistPNG_DSR) ReadTesting(dirname string) ([]gone.Sample, error) {
	return mnist_png.ReadTesting(dirname)
}

var (
	// dsr DataSetReader = mnistDSR{}
	dsr DataSetReader = mnistPNG_DSR{}
)
