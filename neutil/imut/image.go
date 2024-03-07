package imut

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io/ioutil"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
)

func SaveImagePNG(m image.Image, filename string) error {
	var b bytes.Buffer
	err := png.Encode(&b, m)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, b.Bytes(), 0666)
}

func MakeGray(m image.Image) *image.Gray {
	if g, ok := m.(*image.Gray); ok {
		return g
	}
	r := m.Bounds()
	g := image.NewGray(r)
	draw.Draw(g, r, m, image.ZP, draw.Src)
	return g
}

func MakeSamplesGray(g *image.Gray) []neural.Sample {

	r := g.Bounds()

	size := image.Point{
		X: r.Dx(),
		Y: r.Dy(),
	}

	samples := make([]neural.Sample, 0, size.X*size.Y)

	for x := r.Min.X; x < r.Max.X; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {

			cg := g.GrayAt(x, y)

			var (
				xf = float64(x-r.Min.X) / float64(size.X)
				yf = float64(y-r.Min.Y) / float64(size.Y)
			)

			out := float64(cg.Y) / 255

			sample := neural.Sample{
				Inputs:  []float64{xf, yf},
				Outputs: []float64{out},
			}

			samples = append(samples, sample)
		}
	}

	return samples
}

func MakeGrayFromNeural_outs(p *neural.Neural, size image.Point) *image.Gray {
	var (
		r = image.Rectangle{Max: size}
		g = image.NewGray(r)
	)
	var (
		inputs  = make([]float64, 2)
		outputs = make([]float64, 2)
	)
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {

			var (
				norm_X = float64(x) / float64(size.X-1) // [0..1)
				norm_Y = float64(y) / float64(size.Y-1) // [0..1)
			)
			inputs[0] = norm_X
			inputs[1] = norm_Y

			p.SetInputs(inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			g.Set(x, y, color.Gray{
				Y: uint8(neutil.IndexOfMax(outputs) * 255),
			})
		}
	}
	return g
}

func MakeGrayFromNeural(p *neural.Neural, size image.Point) *image.Gray {
	var (
		r = image.Rectangle{Max: size}
		g = image.NewGray(r)
	)
	var (
		inputs  = make([]float64, 2)
		outputs = make([]float64, 1)
	)
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {

			var (
				norm_X = float64(x) / float64(size.X-1) // [0..1)
				norm_Y = float64(y) / float64(size.Y-1) // [0..1)
			)
			inputs[0] = norm_X
			inputs[1] = norm_Y

			p.SetInputs(inputs)
			p.Calculate()
			p.GetOutputs(outputs)

			g.Set(x, y, color.Gray{
				Y: uint8(outputs[0] * 255),
			})
		}
	}
	return g
}
