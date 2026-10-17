package neutil

// mean squared error
// https://en.wikipedia.org/wiki/Mean_squared_error

// MSE computes the Mean Squared Error between two slices of float64 values.
// The function operates on the intersection of both lengths (safe slice boundaries).
// Formula: MSE = (1/n) * Sum((a[i] - b[i])^2)
func MSE(a, b []float64) float64 {

	n := len(a)
	mustHaveMin("MSE", "length", n, 1)

	if len(b) != n {
		panic(errorHaveWant("MSE", "b length", len(b), n))
	}

	// BCE hints: eliminates bounds checks for both slices in the loop below.
	_ = b[n-1]

	sum := 0.0
	for i := range n {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return sum / float64(n)
}
