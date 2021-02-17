package mnist

import (
	"encoding/binary"
	"fmt"
	"image"
	"io"

	"github.com/gitchander/neural"
)

// http://yann.lecun.com/exdb/mnist

const (
	imagesMagic = 2051
	labelsMagic = 2049
)

//var errInvalidFile = errors.New("invalid file format")

func errInvalidMagic(prefix string, haveMagic, wantMagic uint32) error {
	return fmt.Errorf("%s invalid magic number: have %x, want %x",
		prefix, haveMagic, wantMagic)
}

func errInvalidLabel(label uint8) error {
	return fmt.Errorf("invalid label value %d", label)
}

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
		return nil, errInvalidMagic("mnist.ReadImages:", h.Magic, imagesMagic)
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
		return nil, errInvalidMagic("mnist.ReadLabels:", h.Magic, labelsMagic)
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
		return nil, errInvalidMagic("mnist.ReadInputs:", h.Magic, imagesMagic)
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
		return nil, errInvalidMagic("mnist.ReadOutputs:", h.Magic, labelsMagic)
	}

	labels := make([]uint8, h.NumberOfItems)
	_, err = io.ReadFull(r, labels)
	if err != nil {
		return nil, err
	}

	var ssv = make([][]float64, len(labels))
	for i, label := range labels {
		if (0 <= label) && (label <= 9) {
			ssv[i] = neural.OneHot(10, int(label))
		} else {
			return nil, errInvalidLabel(label)
		}
	}

	return ssv, err
}
