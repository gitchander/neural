package neutil

import (
	"github.com/gitchander/minmax"
)

type float64Slice []float64

func (v float64Slice) Len() int           { return len(v) }
func (v float64Slice) Less(i, j int) bool { return v[i] < v[j] }

func IndexOfMin(vs []float64) (min int) {
	return minmax.IndexOfMin(float64Slice(vs))
}

func IndexOfMax(vs []float64) (max int) {
	return minmax.IndexOfMax(float64Slice(vs))
}
