package neural

// loss function
// cost function

type CostFunc interface {
	Func(t, x []float64) float64 // t - ideal value
	Derivative(ti, xi float64) float64
}

type costMSE struct{}

func (costMSE) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {
		delta := t[i] - x[i]
		sum += delta * delta
	}
	return sum / 2
}

func (costMSE) Derivative(ti, xi float64) float64 {
	delta := ti - xi
	return -delta
}

//func (costMSE) Func(t, x []float64) float64 {
//	var sum float64
//	for i := range t {
//		delta := x[i] - t[i]
//		sum += delta * delta
//	}
//	return sum / 2
//}

//func (costMSE) Derivative(ti, xi float64) float64 {
//	delta := xi - ti
//	return delta
//}
