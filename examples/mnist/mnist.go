package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
)

// http://yann.lecun.com/exdb/mnist

type ImagesHeader struct {
	Magic           uint32 // = 2051
	NumberOfImages  uint32
	NumberOfRows    uint32
	NumberOfColumns uint32
}

type LabelsHeader struct {
	Magic         uint32 // = 2049
	NumberOfItems uint32
}

func WalkMNIST(nameImages, nameLabels string, f func(size image.Point, data []byte, label byte) bool) error {
	fileImages, err := os.Open(nameImages)
	if err != nil {
		return err
	}
	defer fileImages.Close()

	fileLabels, err := os.Open(nameLabels)
	if err != nil {
		return err
	}
	defer fileLabels.Close()

	var (
		images = bufio.NewReader(fileImages)
		labels = bufio.NewReader(fileLabels)
	)

	var imh ImagesHeader
	err = binary.Read(images, binary.BigEndian, &imh)
	if err != nil {
		return err
	}
	fmt.Printf("%+v\n", imh)

	var lbh LabelsHeader
	err = binary.Read(labels, binary.BigEndian, &lbh)
	if err != nil {
		return err
	}
	fmt.Printf("%+v\n", lbh)

	if imh.NumberOfImages != lbh.NumberOfItems {
		return fmt.Errorf("number of images and labels are not equal: %d != %d",
			imh.NumberOfImages, lbh.NumberOfItems)
	}

	size := image.Point{
		X: int(imh.NumberOfColumns),
		Y: int(imh.NumberOfRows),
	}
	imageData := make([]byte, size.X*size.Y)
	n := int(imh.NumberOfImages)
	for i := 0; i < n; i++ {
		_, err = io.ReadFull(images, imageData)
		if err != nil {
			return err
		}
		label, err := labels.ReadByte()
		if err != nil {
			return err
		}
		if !f(size, imageData, label) {
			break
		}
		//fmt.Println(i, label)
	}

	return nil
}

func imageFromData(size image.Point, data []byte) (*image.Gray, error) {
	if len(data) < size.X*size.Y {
		return nil, errors.New("insufficient data length")
	}
	g := image.NewGray(image.Rectangle{Max: size})
	for y := 0; y < size.Y; y++ {
		for x := 0; x < size.X; x++ {
			g.Set(x, y, color.Gray{Y: data[0]})
			data = data[1:]
		}
	}
	return g, nil
}
