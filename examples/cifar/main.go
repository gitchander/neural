package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"os"
)

// https://www.cs.toronto.edu/~kriz/cifar.html

func main() {
	err := readDataBatch("cifar-10-batches-bin/data_batch_1.bin")
	if err != nil {
		log.Fatal(err)
	}
}

type Interval struct {
	Min, Max int
}

func readDataBatch(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	const (
		labelSize     = 1
		channelsCount = 3
	)
	var (
		size       = image.Point{X: 32, Y: 32}
		sampleSize = labelSize + (size.X * size.Y * channelsCount)
	)
	in := Interval{Min: 9800, Max: 10000}

	_, err = file.Seek(int64(sampleSize)*int64(in.Min), os.SEEK_SET)
	if err != nil {
		return err
	}

	r := bufio.NewReader(file)

	sampleData := make([]byte, sampleSize)

	for i := in.Min; i < in.Max; i++ {
		_, err = io.ReadFull(r, sampleData)
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		label := sampleData[0]
		fmt.Println("label", label)

		im, err := imageFromData(sampleData[1:], size)
		if err != nil {
			return err
		}

		err = imageSavePNG(im, fmt.Sprintf("images/sample-%06d.png", i))
		if err != nil {
			return err
		}
	}
	return nil
}

func imageFromData(bs []byte, size image.Point) (image.Image, error) {

	im := image.NewRGBA(image.Rectangle{Max: size})

	var (
		shiftChan = size.X * size.Y

		shift_A = shiftChan * 0
		shift_G = shiftChan * 1
		shift_B = shiftChan * 2
	)

	for y := 0; y < size.Y; y++ {
		for x := 0; x < size.X; x++ {
			c := color.RGBA{
				R: bs[shift_A],
				G: bs[shift_G],
				B: bs[shift_B],
				A: 255,
			}
			im.Set(x, y, c)
			bs = bs[1:]
		}
	}
	return im, nil
}

func imageSavePNG(im image.Image, filename string) error {
	var buf bytes.Buffer
	err := png.Encode(&buf, im)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, buf.Bytes(), 0664)
}
