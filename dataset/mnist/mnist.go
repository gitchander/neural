package mnist

import (
	"encoding/binary"
	"errors"
	"image"
	"io"

	"github.com/gitchander/neural"
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

func ReadImages(r io.Reader) ([]*image.Gray, error) {
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

func ReadLabels(r io.Reader) ([]uint8, error) {
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

func ReadInputs(r io.Reader) ([][]float64, error) {
	var h imagesHeader
	err := binary.Read(r, binary.BigEndian, &h)
	if err != nil {
		return nil, err
	}

	if h.Magic != imagesMagic {
		return nil, errInvalidFile
	}

	var (
		ssv = make([][]float64, h.NumberOfImages)

		n   = h.NumberOfColumns * h.NumberOfRows
		buf = make([]byte, n)
	)
	for i := range ssv {
		_, err := io.ReadFull(r, buf)
		if err != nil {
			return nil, err
		}
		sv := make([]float64, n)
		for j, b := range buf {
			sv[j] = float64(b) / 255
		}
		ssv[i] = sv
	}

	return ssv, nil
}

func ReadOutputs(r io.Reader) ([][]float64, error) {
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
	if err != nil {
		return nil, err
	}

	var ssv = make([][]float64, len(labels))
	for i, label := range labels {
		ssv[i] = neural.OneHot(10, int(label))
	}

	return ssv, err
}
