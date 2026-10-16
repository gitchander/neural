package neutil

// mean squared error
// https://en.wikipedia.org/wiki/Mean_squared_error

// MSE computes the Mean Squared Error between two slices of float64 values.
// The function operates on the intersection of both lengths (safe slice boundaries).
// Formula: MSE = (1/n) * Sum((a[i] - b[i])^2)
func MSE(a, b []float64) float64 {
	n := len(a)
	if n != len(b) {
		panic("length not equal")
	}
	sum := 0.0
	// Utilizing range n leverages the Go compiler's BCE optimization,
	// completely avoiding implicit array boundary checks inside the loop.
	for i := range n {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return sum / float64(n)
}
