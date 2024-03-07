package imut

import (
	"image"
)

var _ image.Point = image.Pt(0, 0)

type Point2f struct {
	X, Y float64
}

// Pt2f is shorthand for [Point2f]{X, Y}.
func Pt2f(x, y float64) Point2f {
	return Point2f{
		X: x,
		Y: y,
	}
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
