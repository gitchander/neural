package neutil

import (
	"fmt"
	"math"
)

func checkFloat64IsReal(v float64) error {
	if math.IsNaN(v) {
		return fmt.Errorf("value is NaN")
	}
	if math.IsInf(v, -1) {
		return fmt.Errorf("value is -Inf")
	}
	if math.IsInf(v, +1) {
		return fmt.Errorf("value is +Inf")
	}
	return nil
}

const logEpsilon = 1e-15

func safeLog(x float64) float64 {
	if x < logEpsilon {
		x = logEpsilon
	}
	y := math.Log(x)
	err := checkFloat64IsReal(y)
	if err != nil {
		panic(err)
	}
	return y
}
