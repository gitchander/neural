package neutil

func Mean(xs []float64) float64 {
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

// mean squared error
// https://en.wikipedia.org/wiki/Mean_squared_error
func MSE(a, b []float64) float64 {
	n := len(a)
	if n != len(b) {
		panic("length not equal")
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return sum / float64(n)
}

type Statistics struct {
	sum    float64
	sumSqr float64
	n      int
}

func (st *Statistics) Reset() {
	st.sum = 0
	st.sumSqr = 0
	st.n = 0
}

func (st *Statistics) Add(x float64) {
	st.sum += x
	st.sumSqr += x * x
	st.n++
}

func (st *Statistics) Mean() float64 {
	return st.sum / float64(st.n)
}

func (st *Statistics) MSE() float64 {
	return st.sumSqr / float64(st.n)
}
