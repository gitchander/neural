package neutil

import (
	"github.com/gitchander/neural"
)

func MakeOutputs(n, i int) []float64 {
	outputs := make([]float64, n)
	outputs[i] = 1
	return outputs
}

func NormalizationInputs(samples []neural.Sample) {

	var (
		firstSample = samples[0]

		n   = len(firstSample.Inputs)
		min = make([]float64, n)
		max = make([]float64, n)
	)

	for i, v := range firstSample.Inputs {
		min[i] = v
		max[i] = v
	}

	for k := 1; k < len(samples); k++ {
		sample := samples[k]
		for i, v := range sample.Inputs {
			if v < min[i] {
				min[i] = v
			}
			if v > max[i] {
				max[i] = v
			}
		}
	}

	for _, sample := range samples {
		var vs = sample.Inputs
		for i := range vs {
			vs[i] = normal(vs[i], min[i], max[i])
		}
	}
}

func normal(x, min, max float64) float64 {
	return (x - min) / (max - min)
}
