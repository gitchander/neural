package mnist

import (
	"errors"
	"image"

	"github.com/gitchander/neural"
)

func MakeSamples(nameImages, nameLabels string) ([]neural.Sample, error) {

	images, err := ReadImagesFile(nameImages)
	if err != nil {
		return nil, err
	}

	labels, err := ReadLabelsFile(nameLabels)
	if err != nil {
		return nil, err
	}

	n := len(images)
	if n != len(labels) {
		return nil, errors.New("number of images not equal number of labels")
	}

	samples := make([]neural.Sample, n)
	for i := 0; i < n; i++ {
		samples[i] = neural.Sample{
			Inputs:  InputsFromImage(images[i]),
			Outputs: OutputsFromLabel(labels[i]),
		}
	}
	return samples, nil
}

func InputsFromImage(g *image.Gray) (inputs []float64) {
	inputs = make([]float64, len(g.Pix))
	for i, p := range g.Pix {
		inputs[i] = float64(p) / 255
	}
	return inputs
}

func OutputsFromLabel(label byte) (outputs []float64) {
	outputs = make([]float64, 10)
	outputs[label] = 1
	return outputs
}
