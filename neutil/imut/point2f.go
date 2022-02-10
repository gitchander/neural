package imut

import "math"

type Point2f struct {
	X, Y float64
}

func (a Point2f) Add(b Point2f) Point2f {
	return Point2f{
		X: a.X + b.X,
		Y: a.Y + b.Y,
	}
}

func (a Point2f) Sub(b Point2f) Point2f {
	return Point2f{
		X: a.X - b.X,
		Y: a.Y - b.Y,
	}
}

func (a Point2f) MulScalar(scalar float64) Point2f {
	return Point2f{
		X: a.X * scalar,
		Y: a.Y * scalar,
	}
}

func (a Point2f) DivScalar(scalar float64) Point2f {
	return Point2f{
		X: a.X / scalar,
		Y: a.Y / scalar,
	}
}

//------------------------------------------------------------------------------
// https://brilliant.org/wiki/convert-polar-coordinates-to-cartesian/
//------------------------------------------------------------------------------
// x = r * cos(θ)
// y = r * sin(θ)​
//------------------------------------------------------------------------------
func PolarToCartesian(radius, angle float64) Point2f {
	sin, cos := math.Sincos(angle)
	p := Point2f{X: cos, Y: sin}
	return p.MulScalar(radius)
}
