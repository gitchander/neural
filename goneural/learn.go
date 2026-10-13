package goneural

import (
	"fmt"

	"github.com/gitchander/neural/neutil/random"
)

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

//------------------------------------------------------------------------------

func ShuffleSamples(samples []Sample) {
	r := random.NewRandNow()
	r.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
}

var ShuffleSamplesFunc = func(samples []Sample) func() {
	r := random.NewRandNow()
	return func() {
		r.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
	}
}

//------------------------------------------------------------------------------

// Learn
func Train(p *Neural, samples []Sample, learnRate float64, epochMax int,
	f func(epoch int, averageCost float64) bool) error {

	err := checkSamplesTopology(p, samples)
	if err != nil {
		return err
	}

	bp := NewBP(p, nil)
	bp.SetLearningRate(learnRate)
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost := bp.TrainSamples(samples)
		if !f(epoch, averageCost) {
			break
		}
	}

	return nil
}

func TestSamples(p *Neural, samples []Sample) (*TestResult, error) {
	err := checkSamplesTopology(p, samples)
	if err != nil {
		return nil, err
	}
	bp := NewBP(p, nil)
	result := bp.TestSamples(samples)
	return &result, nil
}

func checkSamplesTopology(p *Neural, samples []Sample) error {

	format := "sample %d invalid %s length: have %d, want %d"

	var (
		inLen  = len(p.getInputLayer().neurons)
		outLen = len(p.getOutputLayer().neurons)
	)

	for i, sample := range samples {
		if len(sample.Inputs) != inLen {
			return fmt.Errorf(format, i, "inputs", len(sample.Inputs), inLen)
		}
		if len(sample.Outputs) != outLen {
			return fmt.Errorf(format, i, "outputs", len(sample.Outputs), outLen)
		}
	}

	return nil
}
