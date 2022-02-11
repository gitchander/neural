package neural

import (
	"fmt"
)

type Sample struct {
	Inputs  []float64
	Outputs []float64
}

func Learn(p *Neural, samples []Sample, learnRate float64, epochMax int,
	f func(epoch int, averageCost float64) bool) error {

	err := checkSamplesTopology(p, samples)
	if err != nil {
		return err
	}

	bp := NewBP(p, CFMeanSquared)
	bp.SetLearningRate(learnRate)
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost := bp.LearnSamples(samples)
		if !f(epoch, averageCost) {
			break
		}
	}

	return nil
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
