package goneural

import (
	"io"

	"github.com/gitchander/neural/neutil/baserw"
)

func Encode(w io.Writer, p *Neural) error {

	bw := baserw.NewBaseWriter(w)

	err := bw.WriteCompactInt(len(p.layers))
	if err != nil {
		return err
	}
	for _, l := range p.layers {
		err = writeLayerConfig(bw, l.lc)
		if err != nil {
			return err
		}
	}

	for k := 1; k < len(p.layers); k++ {
		l := p.layers[k]
		for _, n := range l.neurons {
			for _, weight := range n.weights {
				if err = bw.WriteFloat64(weight); err != nil {
					return err
				}
			}
			if err = bw.WriteFloat64(n.bias); err != nil {
				return err
			}
		}
	}

	return nil
}

func Decode(r io.Reader) (*Neural, error) {

	br := baserw.NewBaseReader(r)

	n, err := br.ReadCompactInt()
	if err != nil {
		return nil, err
	}
	lcs := make([]LayerConfig, n)
	for i := range lcs {
		lc, err := readLayerConfig(br)
		if err != nil {
			return nil, err
		}
		lcs[i] = *lc
	}

	p, err := NewNeural(lcs)
	if err != nil {
		return nil, err
	}

	for k := 1; k < len(p.layers); k++ {
		l := p.layers[k]
		for _, n := range l.neurons {
			for i := range n.weights {
				v, err := br.ReadFloat64()
				if err != nil {
					return nil, err
				}
				n.weights[i] = v
			}
			v, err := br.ReadFloat64()
			if err != nil {
				return nil, err
			}
			n.bias = v
		}
	}

	return p, nil
}
