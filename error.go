package neural

type errorFunc struct{}

// t - ideal value
func (errorFunc) Func(t, x float64) float64 {
	delta := t - x
	return delta * delta / 2
}

func (errorFunc) Derivative(t, x float64) float64 {
	return x - t
}

var errFunc = errorFunc{}
