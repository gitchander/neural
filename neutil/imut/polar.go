package imut

import (
	"math"
)

// https://en.wikipedia.org/wiki/Polar_coordinate_system
// https://brilliant.org/wiki/convert-polar-coordinates-to-cartesian/

// Rho
// https://en.wikipedia.org/wiki/Rho
// letters: Ρ, ρ

// Phi
// https://en.wikipedia.org/wiki/Phi
// letters: θ, φ

type Polar struct {
	Rho float64
	Phi float64
}

// x = r * cos(θ)
// y = r * sin(θ)​

func PolarToCartesian(p Polar) Point2f {
	sin, cos := math.Sincos(p.Phi)
	c := Point2f{
		X: cos,
		Y: sin,
	}
	c = c.MulScalar(p.Rho)
	return c
}
