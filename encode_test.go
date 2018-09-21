package neural

import (
	"bytes"
	"testing"
)

func TestEncode(t *testing.T) {
	p, err := NewMLP(7, 100, 10)
	if err != nil {
		t.Fatal(err)
	}
	p.RandomizeWeights()

	var buf bytes.Buffer
	err = Encode(&buf, p)
	if err != nil {
		t.Fatal(err)
	}

	bs := buf.Bytes()

	t.Logf("len bytes %d", len(bs))
	//t.Logf("%x", bs)

	q, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if !Equal(p, q) {
		t.Fatal("not equal")
	}
}

func Equal(a, b *MLP) bool {
	var (
		layersA = a.layers
		layersB = b.layers
	)
	if len(layersA) != len(layersB) {
		return false
	}
	for k := range layersA {
		var (
			nsA = layersA[k].ns
			nsB = layersB[k].ns
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
