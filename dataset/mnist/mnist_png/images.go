package mnist_png

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"path/filepath"

	_ "image/png"

	gone "github.com/gitchander/neural/goneural"
	"github.com/gitchander/neural/neutil/imut"
	"github.com/gitchander/neural/neutil/osutil"
)

func MakeImages(dirname string, samples []gone.Sample) error {
	err := osutil.MkdirIfNotExist(dirname)
	if err != nil {
		return err
	}
	m := make(map[int][][]float64)
	for _, sample := range samples {
		digit := gone.IndexOfMax(sample.Outputs)
		fs := m[digit]
		fs = append(fs, sample.Inputs)
		m[digit] = fs
	}
	imageSize := image.Pt(28, 28)
	for digit, fs := range m {
		dir := filepath.Join(dirname, formatInt(digit))
		err := osutil.MkdirIfNotExist(dir)
		if err != nil {
			return err
		}
		for i, inputs := range fs {
			filename := filepath.Join(dir, fmt.Sprintf("%d.png", i))
			err := makeImagePNG(filename, imageSize, inputs)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func makeImagePNG(filename string, imageSize image.Point, samples []float64) error {

	sizeXY := imageSize.X * imageSize.Y
	if len(samples) != sizeXY {
		return fmt.Errorf("invalid samples size: have %d, want %d", len(samples), sizeXY)
	}

	r := image.Rect(0, 0, imageSize.X, imageSize.Y)

	coordOffset := func(c image.Point) int {
		d := c.Sub(r.Min)
		return (d.Y * imageSize.X) + d.X
	}

	m := image.NewGray(r)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			var (
				offset = coordOffset(image.Pt(x, y))
				v      = samples[offset]
			)
			if v < 0 {
				return fmt.Errorf("v < 0")
			}
			if v > 1 {
				return fmt.Errorf("v > 1")
			}
			cl := color.Gray{
				Y: uint8(math.Round(float64(v * 255))),
			}
			m.Set(x, y, cl)
		}
	}

	return imut.SaveImagePNG(m, filename)
}
