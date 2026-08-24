package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/gitchander/neural/dataset/mnist/mnist_gz"
	"github.com/gitchander/neural/dataset/mnist/mnist_png"
	gone "github.com/gitchander/neural/goneural"
	"github.com/gitchander/neural/neutil"
	"github.com/gitchander/neural/neutil/random"
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

	r := random.NewRandNow()
	shuffleSamples := func() {
		r.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
	}
	shuffleSamples()

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
		//learnRate = 0.6
		learnRate = 0.2
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

	err = gone.Learn(p, samples, learnRate, epochMax, f)
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

	r := random.NewRandNow()
	shuffleSamples := func() {
		r.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
	}
	shuffleSamples()

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

// func runTest2(dirname, neuralName string) error {

// 	//samples, err := dsr.ReadTraining(dirname)
// 	samples, err := dsr.ReadTesting(dirname)
// 	if err != nil {
// 		return err
// 	}

// 	p, err := gone.ReadFile(neuralName)
// 	if err != nil {
// 		return err
// 	}

// 	fmt.Println("topology:", p.Topology())

// 	outputs := make([]float64, 10)

// 	cf := gone.CrossEntropy{}
// 	var sum float64
// 	for _, sample := range samples {

// 		p.SetInputs(sample.Inputs)
// 		p.Calculate()
// 		p.GetOutputs(outputs)

// 		//fmt.Println("sum Outputs:", sample.Outputs)
// 		//fmt.Println(outputs, sample.Outputs)

// 		sum += cf.Func(sample.Outputs, outputs)
// 	}
// 	averageCost := sum / float64(len(samples))

// 	fmt.Printf("average cost = %.10f\n", averageCost)

// 	return nil
// }

func sumFloat64s(as []float64) float64 {
	sum := 0.0
	for _, a := range as {
		sum += a
	}
	return sum
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
