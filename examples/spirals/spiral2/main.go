package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"

	"github.com/fogleman/gg"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil/imut"
)

// https://www.youtube.com/watch?v=p_kckXJUxxg
// https://www.youtube.com/watch?v=PRSva4fIkXA

func main() {
	err := doSpiral()
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func fillContext(dc *gg.Context, c color.Color) {
	dc.SetFillStyle(gg.NewSolidPattern(c))
	dc.Clear()
}

func doSpiral() error {
	samples := makeSamples()
	size := image.Point{X: 512, Y: 512}

	//-----------------------------------------------
	if false {
		dc := gg.NewContext(size.X, size.Y)
		for _, sample := range samples {
			var (
				x = sample.Inputs[0] * float64(size.X)
				y = sample.Inputs[1] * float64(size.Y)
			)
			dc.DrawCircle(x, y, 10)
			if sample.Outputs[0] < 0.5 {
				dc.SetRGB(0, 0, 0)
			} else {
				dc.SetRGB(1, 1, 1)
			}
			dc.Fill()
		}
		return dc.SavePNG("out.png")
	}
	//-----------------------------------------------

	p, err := neural.NewMLP(2, 100, 1)
	//p, err := neural.NewMLP(2, 20, 20, 1)
	if err != nil {
		return err
	}
	p.RandomizeWeights()

	const (
		learnRate = 0.8
		epochMax  = 1000000
		epsilon   = 0.000001
	)

	filename := "neuro-spiral.png"

	f := func(epoch int, averageCost float64) bool {
		fmt.Println("epoch:", epoch)
		fmt.Printf("average cost: %.7f\n", averageCost)

		if (epoch > 0) && ((epoch % 1000) == 0) {
			err = makeOutImage(p, size, samples, filename)
			if err != nil {
				return false
			}
		}

		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	if err != nil {
		return err
	}

	return makeOutImage(p, size, samples, filename)
}

func makeOutImage(p *neural.MLP, size image.Point, samples []neural.Sample, filename string) error {

	g := imut.MakeGrayFromMLP(p, size)

	dc := gg.NewContext(size.X, size.Y)
	dc.DrawImage(g, 0, 0)

	dx := float64(size.X)
	dy := float64(size.Y)

	for _, sample := range samples {
		x := sample.Inputs[0] * dx
		y := sample.Inputs[1] * dy
		out := sample.Outputs[0]

		dc.DrawCircle(x, y, 3)

		//--------------------------------------
		//		dc.SetRGB(out, out, out)
		//		dc.FillPreserve()
		//		if out > 0.5 {
		//			dc.SetRGB(1, 0, 0)
		//		} else {
		//			dc.SetRGB(0, 0.5, 0)
		//		}
		//		dc.Stroke()
		//--------------------------------------
		//		if out < 0.5 {
		//			dc.SetRGB(1, 1, 1)
		//		} else {
		//			dc.SetRGB(0, 0, 0)
		//		}
		if out < 0.5 {
			dc.SetRGB(0, 0.5, 0)
		} else {
			dc.SetRGB(1, 0, 0)
		}
		dc.Stroke()
		//--------------------------------------
	}

	return dc.SavePNG(filename)
	//err = SaveImagePNG(, filename)
}

func makeSamples() []neural.Sample {

	center := imut.Point2f{X: 1, Y: 1}.DivScalar(2)

	angle := math.Pi * 0.3
	angleFactor := (1 / math.Pi) * 0.08
	angleDelta := 0.25

	n := 140
	samples := make([]neural.Sample, 2*n)

	for i := 0; i < n; i++ {

		R := angle * angleFactor

		p := center.Add(imut.PolarToCartesian(R, angle))

		samples[i*2+0] = neural.Sample{
			Inputs:  []float64{p.X, p.Y},
			Outputs: []float64{0},
		}

		p = center.Add(imut.PolarToCartesian(R, angle+math.Pi))

		samples[i*2+1] = neural.Sample{
			Inputs:  []float64{p.X, p.Y},
			Outputs: []float64{1},
		}

		angle += angleDelta
		angleDelta *= 0.987
	}

	return samples
}

func makeSamplesGrayN(g *image.Gray, n int, r *rand.Rand) []neural.Sample {

	bounds := g.Bounds()

	dx := float64(bounds.Dx())
	dy := float64(bounds.Dy())

	samples := make([]neural.Sample, 0, n)

	for i := 0; i < n; i++ {

		xf := r.Float64()
		yf := r.Float64()

		x := bounds.Min.X + int(math.Round(xf*dx))
		y := bounds.Min.Y + int(math.Round(yf*dy))

		cg := g.GrayAt(x, y)

		//		xf := float64(x-r.Min.X) / float64(dx)
		//		yf := float64(y-r.Min.Y) / float64(dy)

		out := float64(cg.Y) / 255

		sample := neural.Sample{
			Inputs:  []float64{xf, yf},
			Outputs: []float64{out},
		}

		samples = append(samples, sample)
	}

	return samples
}
