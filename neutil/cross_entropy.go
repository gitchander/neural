package neutil

//------------------------------------------------------------------------------

// https://www.parasdahal.com/softmax-crossentropy

// Bounds check elimination (BCE) hint.
// $ go build -gcflags="-d=ssa/check_bce/debug=1" cross_entropy.go utils.go

//------------------------------------------------------------------------------

type crossEntropy struct{}

func (crossEntropy) Func(t, x []float64) float64 {

	n := len(t)
	if len(x) != n {
		panic(errorHaveWant("crossEntropy.Func", "x length", len(x), n))
	}

	// BCE hint: eliminates bounds checks in the loops below.
	_ = x[n-1]

	var sum float64
	for i := range n {
		sum += t[i] * safeLog(x[i])
	}
	return -sum
}

// Derivative of Cross Entropy Loss with Softmax
func (crossEntropy) SoftmaxDerivative(ti, xi float64) float64 {
	return xi - ti
}

var CrossEntropy = crossEntropy{}

//------------------------------------------------------------------------------

type binaryCrossEntropy struct{}

func (binaryCrossEntropy) Func(t, x []float64) float64 {

	n := len(t)
	if len(x) != n {
		panic(errorHaveWant("binaryCrossEntropy.Func", "x length", len(x), n))
	}

	// BCE hint: eliminates bounds checks in the loops below.
	_ = x[n-1]

	var sum float64
	for i := range n {

		// if (t[i] == 1) -> -log(x[i])
		// if (t[i] == 0) -> -log(1-x[i])

		sum += (t[i] * safeLog(x[i])) + ((1 - t[i]) * safeLog(1-x[i]))
	}
	return -sum
}

// Derivative of Binary Cross Entropy Loss with Sigmoid
func (binaryCrossEntropy) SigmoidDerivative(ti, xi float64) float64 {
	return (xi - ti) / (xi * (1 - xi))
}

var BinaryCrossEntropy = binaryCrossEntropy{}

//------------------------------------------------------------------------------
