package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"image"
	"io"
	"os"
)

// http://yann.lecun.com/exdb/mnist

const (
	imagesMagic = 2051
	labelsMagic = 2049
)

var errInvalidFile = errors.New("invalid file format")

type imagesHeader struct {
	Magic           uint32
	NumberOfImages  uint32
	NumberOfRows    uint32
	NumberOfColumns uint32
}

type labelsHeader struct {
	Magic         uint32
	NumberOfItems uint32
}

func ReadImagesFile(filename string) ([]*image.Gray, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	zr, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	return readImages(zr)
}

func readImages(r io.Reader) ([]*image.Gray, error) {
	var h imagesHeader
	err := binary.Read(r, binary.BigEndian, &h)
	if err != nil {
		return nil, err
	}

	if h.Magic != imagesMagic {
		return nil, errInvalidFile
	}

	size := image.Point{
		X: int(h.NumberOfColumns),
		Y: int(h.NumberOfRows),
	}

	rect := image.Rectangle{Max: size}
	gs := make([]*image.Gray, h.NumberOfImages)
	for i := range gs {
		g := image.NewGray(rect)
		_, err := io.ReadFull(r, g.Pix)
		if err != nil {
			return nil, err
		}
		gs[i] = g
	}

	return gs, nil
}

func ReadLabelsFile(filename string) ([]uint8, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	zr, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	return readLabels(zr)
}

func readLabels(r io.Reader) ([]uint8, error) {
	var h labelsHeader
	err := binary.Read(r, binary.BigEndian, &h)
	if err != nil {
		return nil, err
	}

	if h.Magic != labelsMagic {
		return nil, errInvalidFile
	}

	labels := make([]uint8, h.NumberOfItems)
	_, err = io.ReadFull(r, labels)
	return labels, err
}
