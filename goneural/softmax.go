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

		//xxx := n.out

		n.out = math.Exp(n.out - maxLogit)

		// if minValue := 1e-15; n.out < minValue {
		// 	n.out = minValue
		// }

		// if n.out == 0 {
		// 	fmt.Println(">>", xxx, maxLogit, xxx-maxLogit)
		// }

		sum += n.out
	}

	for _, n := range neurons {
		n.out /= sum
	}

	// for _, n := range neurons {
	// 	if n.out == 0 {
	// 		for _, nn := range neurons {
	// 			fmt.Println(nn.out)
	// 		}
	// 		fmt.Println("sum", sum)
	// 		panic("out = 0")
	// 	}
	// }
}
