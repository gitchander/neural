package mnist

import (
	"errors"
	"image"

	"github.com/gitchander/neural"
)

func MakeSamples(sf SetFiles) ([]neural.Sample, error) {

	inputs, err := ReadInputsFile(sf.Images)
	if err != nil {
		return nil, err
	}

	outputs, err := ReadOutputsFile(sf.Labels)
	if err != nil {
		return nil, err
	}

	n := len(inputs)
	if n != len(outputs) {
		return nil, errors.New("number of inputs not equal number of outputs")
	}

	samples := make([]neural.Sample, n)
	for i := 0; i < n; i++ {
		samples[i] = neural.Sample{
			Inputs:  inputs[i],
			Outputs: outputs[i],
		}
	}
	return samples, nil
}

func InputsFromImage(g *image.Gray) (inputs []float64) {
	inputs = make([]float64, len(g.Pix))
	for i, p := range g.Pix {
		inputs[i] = byteToFloat[p]
	}
	return inputs
}

func OutputsFromLabel(label byte) (outputs []float64) {
	outputs = make([]float64, 10)
	outputs[label] = 1
	return outputs
}
