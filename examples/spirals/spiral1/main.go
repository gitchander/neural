package main

import (
	"fmt"
	"image"
	"log"
	"math"
	"time"

	"github.com/fogleman/gg"
	"github.com/nfnt/resize"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil/imut"
)

func main() {
	err := doSpiral()
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func doSpiral() error {

	size := image.Point{X: 512, Y: 512}

	dc := gg.NewContext(size.X, size.Y)

	//fillContext(dc, color.White)

	drawSpiral(dc, size)

	g := imut.MakeGray(dc.Image())

	err := imut.SaveImagePNG(g, "spiral.png")
	if err != nil {
		return err
	}

	//mr := resize.Resize(50, 50, g, resize.Bicubic)
	mr := resize.Resize(64, 64, g, resize.Lanczos3)

	err = imut.SaveImagePNG(mr, "spiral-samples.png")
	if err != nil {
		return err
	}

	g = imut.MakeGray(mr)

	samples := imut.MakeSamplesGray(g)
	//samples := imut.MakeSamplesGrayN(g, 2000, newRandNow())

	fmt.Println("len samples:", len(samples))

	layers := neural.MakeLayers(neural.ActSigmoid, 2, 50, 50, 1)
	p, err := neural.NewNeural(layers)
	if err != nil {
		return err
	}
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
			err = imut.SaveImagePNG(imut.MakeGrayFromNeural(p, size), filename)
			if err != nil {
				return false
			}
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
	if err != nil {
		return err
	}

	return imut.SaveImagePNG(imut.MakeGrayFromNeural(p, size), filename)
}

func drawSpiral(dc *gg.Context, size image.Point) {

	center := imut.Point2f{
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

		p := center.Add(imut.PolarToCartesian(R, angle))

		dc.DrawCircle(p.X, p.Y, radius)
		dc.SetRGB(0, 0, 0)
		dc.Fill()

		p = center.Add(imut.PolarToCartesian(R, angle+math.Pi))

		dc.DrawCircle(p.X, p.Y, radius)
		dc.SetRGB(1, 1, 1)
		dc.Fill()

		angle += angleDelta
	}
}
