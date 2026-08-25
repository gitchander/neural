package goneural

import (
	"math"
	"slices"
)

func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	var (
		maxLogit = slices.Max(logits) // maxValue
		ys       = make([]float64, len(logits))
	)
	var sum float64
	for i, x := range logits {
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
	for _, n := range neurons {
		if maxLogit < n.out {
			maxLogit = n.out
		}
	}

	var sum float64
	for _, n := range neurons {
		n.out = math.Exp(n.out - maxLogit)
		sum += n.out
	}

	for _, n := range neurons {
		n.out /= sum
	}
}
