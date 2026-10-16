package neutil

//------------------------------------------------------------------------------

// type CrossEntropy struct{}

// func (CrossEntropy) Func(t, x []float64) float64 {
// 	var sum float64
// 	for i := range t {
// 		sum += t[i] * safeLog(x[i])
// 	}
// 	return -sum
// }

// // Derivative of Cross Entropy Loss with Softmax
// func (CrossEntropy) SoftmaxDerivative(ti, xi float64) float64 {
// 	return xi - ti
// }

//------------------------------------------------------------------------------

func CrossEntropy(t, x []float64) (float64, int) {
	n := min(len(t), len(x))
	if n == 0 {
		return 0.0, 0
	}
	var sum float64
	for i := range n {
		sum += t[i] * safeLog(x[i])
	}
	return -sum, n
}

// Derivative of Cross Entropy Loss with Softmax
func CrossEntropySoftmaxDerivative(ti, xi float64) float64 {
	return xi - ti
}

//------------------------------------------------------------------------------

type BinaryCrossEntropy struct{}

func (BinaryCrossEntropy) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {

		// if (t[i] == 1) -> -log(x[i])
		// if (t[i] == 0) -> -log(1-x[i])

		sum -= (t[i] * safeLog(x[i])) + ((1 - t[i]) * safeLog(1-x[i]))
	}
	return sum
}
