package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/gitchander/neural/dataset/mnist/mnist_gz"
	"github.com/gitchander/neural/dataset/mnist/mnist_png"
	gone "github.com/gitchander/neural/goneural"
)

func main() {
	err := run()
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func run() error {

	var c Config

	flag.StringVar(&(c.Dirname), "dir", ".", "mnist directory")
	flag.StringVar(&(c.Mode), "mode", "test", "train or test mode")
	flag.StringVar(&(c.NeuralName), "neural", "neural", "filename neural network")

	flag.Parse()

	return runConfig(c)
}

type Config struct {
	Dirname    string
	Mode       string
	NeuralName string
}

func runConfig(c Config) error {
	switch c.Mode {
	case "train":
		return runTrain(c.Dirname, c.NeuralName)
	case "test":
		return runTest(c.Dirname, c.NeuralName)
	default:
		return fmt.Errorf("invalid mode: %s", c.Mode)
	}
}

func checkSamples(samples []gone.Sample) error {
	for _, sample := range samples {
		for _, v := range sample.Inputs {
			if v < 0 {
				return fmt.Errorf("value (%g) < 0", v)
			}
			if v > 1 {
				return fmt.Errorf("value (%g) > 1", v)
			}
			//fmt.Println(v)
		}
		//fmt.Println(sample.Outputs)

		// for _, v := range sample.Outputs {

		// }
	}
	return nil
}

func runTrain(dirname, neuralName string) error {

	samples, err := dsr.ReadTraining(dirname)
	if err != nil {
		return err
	}

	err = checkSamples(samples)
	if err != nil {
		return err
	}

	p, err := gone.ReadFile(neuralName)
	if err != nil {
		var (
			// layers = []gone.LayerConfig{
			// 	gone.MakeLayerConfigAF((28 * 28), gone.Sigmoid),
			// 	gone.MakeLayerConfigAF((14 * 14), gone.Sigmoid),
			// 	gone.MakeLayerConfigAF(7*7, gone.Sigmoid),
			// 	gone.MakeLayerConfigSoftmax(10),
			// }

			layers = []gone.LayerConfig{
				gone.MakeLayerConfigAF((28 * 28), gone.Sigmoid),
				gone.MakeLayerConfigAF(800, gone.Sigmoid),
				gone.MakeLayerConfigSoftmax(10),
			}
		)
		p, err = gone.NewNeural(layers)
		if err != nil {
			return err
		}
		p.RandomizeWeights()
	}

	shuffleSamples := gone.ShuffleSamplesFunc(samples)
	shuffleSamples()

	const (
		//learnRate = 0.6
		learnRate = 0.1
		epochMax  = 100000
	)

	t0 := time.Now()
	var writeError error

	f := func(epoch int, averageCost float64) bool {

		err := gone.WriteFile(neuralName, p)
		if err != nil {
			writeError = err
			return false
		}

		t1 := time.Now()
		dur := t1.Sub(t0)
		t0 = t1

		fmt.Printf("%s: dur = %v; epoch = %d; average cost = %.10f\n",
			time.Now().Format("15:04:05"),
			dur,
			epoch, averageCost)

		shuffleSamples()

		return true
	}

	err = gone.Train(p, samples, learnRate, epochMax, f)
	if err != nil {
		return err
	}

	return writeError
}

func runTest(dirname, neuralName string) error {

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

	res, err := gone.TestSamples(p, samples)
	if err != nil {
		return err
	}

	fmt.Printf("Average Cost: %.5f\n", res.AverageCost)
	fmt.Printf("Error Rate:   %.3f%%\n", res.ErrorRate)
	fmt.Printf("Accuracy:     %.3f%%\n", res.Accuracy)

	return nil
}

//------------------------------------------------------------------------------

type DataSetReader interface {
	ReadTraining(dirname string) ([]gone.Sample, error)
	ReadTesting(dirname string) ([]gone.Sample, error)
}

type mnistGZ_DSR struct{}

func (mnistGZ_DSR) ReadTraining(dirname string) ([]gone.Sample, error) {
	return mnist_gz.ReadTraining(dirname)
}

func (mnistGZ_DSR) ReadTesting(dirname string) ([]gone.Sample, error) {
	return mnist_gz.ReadTesting(dirname)
}

type mnistPNG_DSR struct{}

func (mnistPNG_DSR) ReadTraining(dirname string) ([]gone.Sample, error) {
	return mnist_png.ReadTraining(dirname)
}

func (mnistPNG_DSR) ReadTesting(dirname string) ([]gone.Sample, error) {
	return mnist_png.ReadTesting(dirname)
}

var (
	//dsr DataSetReader = mnistGZ_DSR{}
	dsr DataSetReader = mnistPNG_DSR{}
)
