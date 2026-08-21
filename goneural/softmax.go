package goneural

import (
	"math"
	"slices"
)

func softmax(xs []float64) []float64 {
	if len(xs) == 0 {
		return nil
	}
	var (
		maxLogit = slices.Max(xs) // maxValue
		ys       = make([]float64, len(xs))
	)
	var sum float64
	for i, x := range xs {
		ys[i] = math.Exp(x - maxLogit)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}

// input logits
func layerSoftmax(lr *layer) {

	neurons := lr.neurons

	if len(neurons) == 0 {
		return
	}

	// The Softmax Function:

	// calc max value:
	maxLogit := neurons[0].out
	for _, currNeuron := range neurons {
		if maxLogit < currNeuron.out {
			maxLogit = currNeuron.out
		}
	}

	var sum float64
	for _, currNeuron := range neurons {
		currNeuron.out = math.Exp(currNeuron.out - maxLogit)
		sum += currNeuron.out
	}

	for _, currNeuron := range neurons {
		currNeuron.out /= sum
	}
}
