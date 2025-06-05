package neutil

import (
	"errors"
	"math"
)

var errNoElements = errors.New("There are no elements")

func square(x float64) float64 {
	return x * x
}

// https://en.wikipedia.org/wiki/Mean
func Mean(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		panic(errNoElements)
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(n)
}

func calcVariance(xs []float64, mean float64) float64 {
	n := len(xs)
	sum := 0.0
	for _, x := range xs {
		sum += square(x - mean)
	}
	return sum / float64(n)
}

func Variance(xs []float64) float64 {
	mean := Mean(xs)
	return calcVariance(xs, mean)
}

// Average RMS: Root Mean Square

type Stat struct {
	Min      float64
	Max      float64
	Mean     float64
	Variance float64
	RMS      float64 // sqrt(Variance)
}

func CalcStat(xs []float64) *Stat {
	min, max := floatsMinMax(xs)
	var (
		mean     = Mean(xs)
		variance = calcVariance(xs, mean)
		rms      = math.Sqrt(variance)
	)
	return &Stat{
		Min:      min,
		Max:      max,
		Mean:     mean,
		Variance: variance,
		RMS:      rms,
	}
}

func floatsMinMax(as []float64) (min, max float64) {
	imin, imax := indexesMinMaxFloat64(as)
	min, max = as[imin], as[imax]
	return
}

func indexesMinMaxFloat64(xs []float64) (imin, imax int) {
	n := len(xs)
	if n == 0 {
		panic(errNoElements)
	}
	imin, imax = 0, 0
	for i := 1; i < n; i++ {
		if xs[imin] > xs[i] {
			imin = i
		}
		if xs[imax] < xs[i] {
			imax = i
		}
	}
	return imin, imax
}
