package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io/ioutil"
	"math"
	"math/rand"
	"time"

	"github.com/fogleman/gg"
	"github.com/gitchander/neural"
	"github.com/nfnt/resize"
)

// https://www.youtube.com/watch?v=p_kckXJUxxg
// https://www.youtube.com/watch?v=PRSva4fIkXA

func main() {
	//spiral1()
	spiral2()
}

func spiral1() {

	size := image.Point{X: 512, Y: 512}

	dc := gg.NewContext(size.X, size.Y)

	//fillContext(dc, color.White)

	drawSpiral(dc, size)

	g := makeGray(dc.Image())

	err := saveImagePNG(g, "spiral.png")
	checkError(err)

	//mr := resize.Resize(50, 50, g, resize.Bicubic)
	mr := resize.Resize(64, 64, g, resize.Lanczos3)

	err = saveImagePNG(mr, "spiral-samples.png")
	checkError(err)

	g = makeGray(mr)

	samples := makeSamplesGray(g)
	//samples := makeSamplesGrayN(g, 2000, newRandNow())

	fmt.Println("len samples:", len(samples))

	p, err := neural.NewMLP(2, 50, 50, 1)
	checkError(err)
	p.RandomizeWeights()

	const (
		learnRate = 0.8
		epochMax  = 100000
		epsilon   = 0.0001
	)

	filename := "neuro-spiral.png"

	start := time.Now()
	f := func(epoch int, averageCost float64) bool {

		if (epoch > 0) && ((epoch % 100) == 0) {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
		}

		t1 := time.Now()
		if t1.Sub(start) > 10*time.Second {
			err = SaveImagePNG(makeGrayFromMLP(p, size), filename)
			checkError(err)

			start = t1
		}

		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)

	err = SaveImagePNG(makeGrayFromMLP(p, size), filename)
	checkError(err)
}

func fillContext(dc *gg.Context, c color.Color) {
	dc.SetFillStyle(gg.NewSolidPattern(c))
	dc.Clear()
}

func drawSpiral(dc *gg.Context, size image.Point) {

	center := Point2f{
		X: float64(size.X),
		Y: float64(size.Y),
	}.DivScalar(2)

	radius := 300.0
	angle := 0.0

	//angleFactor := 2 / math.Pi
	angleFactor := 0.2 / math.Pi
	angleDelta := 0.1

	for i := 0; i < 100; i++ {

		R := radius * (1 + angle*angleFactor)

		p := center.Add(PolarToDecart(R, angle))

		dc.DrawCircle(p.X, p.Y, radius)
		dc.SetRGB(0, 0, 0)
		dc.Fill()

		p = center.Add(PolarToDecart(R, angle+math.Pi))

		dc.DrawCircle(p.X, p.Y, radius)
		dc.SetRGB(1, 1, 1)
		dc.Fill()

		angle += angleDelta
	}
}

func spiral2() {
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
		dc.SavePNG("out.png")
		return
	}
	//-----------------------------------------------

	p, err := neural.NewMLP(2, 100, 1)
	//p, err := neural.NewMLP(2, 20, 20, 1)
	checkError(err)
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
			checkError(err)
		}

		if averageCost < epsilon {
			fmt.Println("epoch:", epoch)
			fmt.Printf("average cost: %.7f\n", averageCost)
			return false
		}
		return true
	}

	err = neural.Learn(p, samples, learnRate, epochMax, f)
	checkError(err)

	err = makeOutImage(p, size, samples, filename)
	checkError(err)
}

func makeOutImage(p *neural.MLP, size image.Point, samples []neural.Sample, filename string) error {

	g := makeGrayFromMLP(p, size)

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

	center := Point2f{X: 1, Y: 1}.DivScalar(2)

	angle := math.Pi * 0.3
	angleFactor := (1 / math.Pi) * 0.08
	angleDelta := 0.25

	n := 140
	samples := make([]neural.Sample, 2*n)

	for i := 0; i < n; i++ {

		R := angle * angleFactor

		p := center.Add(PolarToDecart(R, angle))

		samples[i*2+0] = neural.Sample{
			Inputs:  []float64{p.X, p.Y},
			Outputs: []float64{0},
		}

		p = center.Add(PolarToDecart(R, angle+math.Pi))

		samples[i*2+1] = neural.Sample{
			Inputs:  []float64{p.X, p.Y},
			Outputs: []float64{1},
		}

		angle += angleDelta
		angleDelta *= 0.987
	}

	return samples
}

func makeGray(m image.Image) *image.Gray {
	if g, ok := m.(*image.Gray); ok {
		return g
	}
	r := m.Bounds()
	g := image.NewGray(r)
	draw.Draw(g, r, m, image.ZP, draw.Src)
	return g
}

func saveImagePNG(m image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, m)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0666)
}

func makeSamplesGray(g *image.Gray) []neural.Sample {

	r := g.Bounds()

	dx := r.Dx()
	dy := r.Dy()

	samples := make([]neural.Sample, 0, dx*dy)

	for x := r.Min.X; x < r.Max.X; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {

			cg := g.GrayAt(x, y)

			xf := float64(x-r.Min.X) / float64(dx)
			yf := float64(y-r.Min.Y) / float64(dy)

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

func makeGrayFromMLP_outs(p *neural.MLP, size image.Point) *image.Gray {
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
				Y: uint8(neural.IndexOfMax(outputs) * 255),
			})
		}
	}
	return g
}

func makeGrayFromMLP(p *neural.MLP, size image.Point) *image.Gray {
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

func SaveImagePNG(m image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, m)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0666)
}
