package random

// https://en.wikipedia.org/wiki/Linear_interpolation
// lerp - Linear interpolation
func lerp(v0, v1 float64, t float64) float64 {
	return v0*(1-t) + v1*t
}
