package neural

import (
	"math"
)

// loss function
// cost function

type CostFunc interface {
	Func(t, x []float64) float64 // t - ideal value
	Derivative(ti, xi float64) float64
}

type costMeanSquared struct{}

func (costMeanSquared) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {
		delta := t[i] - x[i]
		sum += delta * delta
	}
	return sum / 2
}

func (costMeanSquared) Derivative(ti, xi float64) float64 {
	delta := ti - xi
	return -delta
}

//func (costMeanSquared) Func(t, x []float64) float64 {
//	var sum float64
//	for i := range t {
//		delta := x[i] - t[i]
//		sum += delta * delta
//	}
//	return sum / 2
//}

//func (costMeanSquared) Derivative(ti, xi float64) float64 {
//	delta := xi - ti
//	return delta
//}

// https://habr.com/post/340792/

type CrossEntropy struct{}

// t[i] - ideal output = [0 or 1]
// x[i] - real output = [0..1]
func (CrossEntropy) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {
		sum -= t[i] * math.Log(x[i])
	}
	return sum
}

func (CrossEntropy) Derivative(ti, xi float64) float64 {
	return xi - ti
}

type BinaryCrossEntropy struct{}

// t[i] - ideal output = [0 or 1]
// x[i] - real output = [0..1]
func (BinaryCrossEntropy) Func(t, x []float64) float64 {
	var sum float64
	for i := range t {

		// if (t[i] == 1) -> -log(x[i])
		// if (t[i] == 0) -> -log(1-x[i])

		sum -= t[i]*math.Log(x[i]) + (1-t[i])*math.Log(1-x[i])
	}
	return sum
}
