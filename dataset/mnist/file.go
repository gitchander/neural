package mnist

import (
	"compress/gzip"
	"image"
	"os"
)

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

	return ReadImages(zr)
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

	return ReadLabels(zr)
}
