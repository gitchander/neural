package goneural

import (
	"bytes"
	"testing"
)

func TestEncode(t *testing.T) {

	lcs := []LayerConfig{
		{
			Activation: ActivationConfig{
				Name: "sigmoid",
			},
			Neurons: 7,
		},
		{
			Activation: ActivationConfig{
				Name: "sigmoid",
			},
			Neurons: 100,
		},
		{
			Activation: ActivationConfig{
				Name: "sigmoid",
			},
			Neurons: 10,
		},
	}

	p, err := NewNeural(lcs)
	if err != nil {
		t.Fatal(err)
	}
	p.RandomizeWeights()

	var buf bytes.Buffer
	err = Encode(&buf, p)
	if err != nil {
		t.Fatal(err)
	}

	//bs := buf.Bytes()
	//t.Logf("len bytes %d", len(bs))

	q, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if !Equal(p, q) {
		t.Fatal("not equal")
	}
}

func Equal(a, b *Neural) bool {
	var (
		layersA = a.layers
		layersB = b.layers
	)
	if len(layersA) != len(layersB) {
		return false
	}
	for k := range layersA {
		var (
			nsA = layersA[k].neurons
			nsB = layersB[k].neurons
		)
		if len(nsA) != len(nsB) {
			return false
		}
		for i := range nsA {
			var (
				nA = nsA[i]
				nB = nsB[i]
			)
			var (
				wsA = nA.weights
				wsB = nB.weights
			)
			if len(wsA) != len(wsB) {
				return false
			}
			for j := range wsA {
				if wsA[j] != wsB[j] {
					return false
				}
			}
			if nA.bias != nB.bias {
				return false
			}
		}
	}
	return true
}
