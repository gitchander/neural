package neutil

import (
	"github.com/gitchander/minmax"
)

type float64Slice []float64

func (v float64Slice) Len() int           { return len(v) }
func (v float64Slice) Less(i, j int) bool { return v[i] < v[j] }

func IndexOfMin(vs []float64) (imin int) {
	return minmax.IndexOfMin(float64Slice(vs))
}

func IndexOfMax(vs []float64) (imax int) {
	return minmax.IndexOfMax(float64Slice(vs))
}

func maxFloat64s(as []float64) float64 {
	if len(as) == 0 {
		panic("There are no elements")
	}
	index := IndexOfMax(as)
	return as[index]
}
