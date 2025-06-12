package mnist_png

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io/fs"
	"io/ioutil"
	"path/filepath"
	"strconv"

	_ "image/png"

	gone "github.com/gitchander/neural/goneural"
)

//------------------------------------------------------------------------------

// https://github.com/myleott/mnist_png

// <training/testing> / <label> / <id>.png

//------------------------------------------------------------------------------

// training
// testing

// database
type DataSet struct {
	Training []gone.Sample // training
	Testing  []gone.Sample // testing
}

func ReadTraining(dirname string) ([]gone.Sample, error) {
	return readSamples(filepath.Join(dirname, "training"))
}

func ReadTesting(dirname string) ([]gone.Sample, error) {
	return readSamples(filepath.Join(dirname, "testing"))
}

func ReadDataSet(dirname string) (*DataSet, error) {

	training, err := ReadTraining(dirname)
	if err != nil {
		return nil, err
	}

	testing, err := ReadTesting(dirname)
	if err != nil {
		return nil, err
	}

	ds := &DataSet{
		Training: training,
		Testing:  testing,
	}

	return ds, nil
}

func serialInts(n int) []int {
	as := make([]int, n)
	for i := range as {
		as[i] = i
	}
	return as
}

func formatInt(a int) string {
	return strconv.Itoa(a)
}

func parseInt(s string) (int, error) {
	return strconv.Atoi(s)
}

type imageConfig struct {
	Format string // "png"
	Size   image.Point
}

var defaultImageConfig = imageConfig{
	Format: "png",
	Size:   image.Pt(28, 28),
}

func readImageInputs(filename string, want imageConfig) ([]float64, error) {

	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	r := bytes.NewReader(data)
	m, imageFormat, err := image.Decode(r)
	if err != nil {
		return nil, err
	}

	if imageFormat != want.Format {
		return nil, fmt.Errorf("Invalid image format: have %s, want %s", imageFormat, want.Format)
	}

	b := m.Bounds()
	size := b.Size()
	if !(size.Eq(want.Size)) {
		return nil, fmt.Errorf("Invalid image size: have %s, want %s", size, want.Size)
	}

	var (
		vs = make([]float64, (size.X * size.Y))

		coordOffset = func(c image.Point) int {
			d := c.Sub(b.Min)
			return (d.Y * size.X) + d.X
		}
	)

	walkBounds(b, func(p image.Point) bool {
		var (
			generalColor = m.At(p.X, p.Y)
			grayColor    = color.GrayModel.Convert(generalColor).(color.Gray)
		)
		vs[coordOffset(p)] = float64(grayColor.Y) / 255 // [0..1]
		return true
	})

	return vs, nil
}

func walkBounds(b image.Rectangle, f func(p image.Point) bool) {
	var (
		y0 = b.Min.Y
		yn = b.Max.Y

		x0 = b.Min.X
		xn = b.Max.X
	)
	for y := y0; y < yn; y++ {
		for x := x0; x < xn; x++ {
			var (
				p  = image.Pt(x, y)
				ok = f(p)
			)
			if !ok {
				return
			}
		}
	}
}

func readSamples(dirname string) ([]gone.Sample, error) {

	var samples []gone.Sample

	digits := serialInts(10) // 0..9

	ic := defaultImageConfig

	for _, digit := range digits {
		var (
			label = formatInt(digit)
			dir   = filepath.Join(dirname, label)
		)

		walkFunc := func(path string, info fs.FileInfo, err error) error {

			if info.IsDir() {
				return nil
			}

			inputs, err := readImageInputs(path, ic)
			if err != nil {
				return err
			}
			sample := gone.Sample{
				Inputs:  inputs,
				Outputs: gone.OneHot(len(digits), digit),
			}
			samples = append(samples, sample)
			return nil
		}

		err := filepath.Walk(dir, walkFunc)
		if err != nil {
			return nil, err
		}
	}
	return samples, nil
}
