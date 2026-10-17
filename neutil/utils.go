package neutil

import (
	"cmp"
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

func errorHaveWant[T comparable](method, name string, have, want T) error {
	return fmt.Errorf("%s: invalid %s: have %v, want %v", method, name, have, want)
}

func mustHaveWant[T comparable](method, name string, have, want T) {
	if have != want {
		panic(errorHaveWant(method, name, have, want))
	}
}

func errorHaveMin[T cmp.Ordered](method, name string, have, minValue T) error {
	return fmt.Errorf("%s: invalid %s: (%v) is less than (%v)", method, name, have, minValue)
}

func mustHaveMin[T cmp.Ordered](method, name string, have, minValue T) {
	if have < minValue {
		panic(errorHaveMin(method, name, have, minValue))
	}
}
