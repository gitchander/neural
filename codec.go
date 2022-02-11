package neural

import (
	"io"

	"github.com/gitchander/neural/neutil/baserw"
)

func Encode(w io.Writer, p *Neural) error {

	bw := baserw.NewBaseWriter(w)

	var err error
	if err = bw.WriteUint16(uint16(len(p.layers))); err != nil {
		return err
	}
	for _, l := range p.layers {
		if err = bw.WriteUint8(uint8(l.at)); err != nil {
			return err
		}
		if err = bw.WriteUint16(uint16(len(l.neurons))); err != nil {
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

	u16, err := br.ReadUint16()
	if err != nil {
		return nil, err
	}
	rs := make([]Layer, int(u16))
	for i := range rs {
		u8, err := br.ReadUint8()
		if err != nil {
			return nil, err
		}
		u16, err := br.ReadUint16()
		if err != nil {
			return nil, err
		}
		rs[i] = Layer{
			ActivationType: ActivationType(u8),
			Neurons:        int(u16),
		}
	}

	p, err := NewNeural(rs)
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
