package neural

import (
	"encoding/binary"
	"io"
	"math"
)

var byteOrder = binary.BigEndian

func Encode(w io.Writer, p *MLP) error {

	bw := newBaseWriter(w)

	var err error
	for _, layer := range p.layers {
		u := uint16(len(layer.ns))
		if err = bw.writeUint16(u); err != nil {
			return err
		}
	}
	if err = bw.writeUint16(0); err != nil {
		return err
	}

	for k := 1; k < len(p.layers); k++ {
		layer := p.layers[k]
		for _, n := range layer.ns {
			for _, weight := range n.weights {
				if err = bw.writeFloat64(weight); err != nil {
					return err
				}
			}
			if err = bw.writeFloat64(n.bias); err != nil {
				return err
			}
		}
	}

	return nil
}

func Decode(r io.Reader) (*MLP, error) {

	br := newBaseReader(r)

	var ds []int
	for {
		u, err := br.readUint16()
		if err != nil {
			return nil, err
		}
		if u == 0 {
			break
		}
		ds = append(ds, int(u))
	}

	p, err := NewMLP(ds...)
	if err != nil {
		return nil, err
	}

	for k := 1; k < len(p.layers); k++ {
		layer := p.layers[k]
		for _, n := range layer.ns {
			for i := range n.weights {
				v, err := br.readFloat64()
				if err != nil {
					return nil, err
				}
				n.weights[i] = v
			}
			v, err := br.readFloat64()
			if err != nil {
				return nil, err
			}
			n.bias = v
		}
	}

	return p, nil
}

type baseWriter struct {
	buf [8]byte
	w   io.Writer
}

func newBaseWriter(w io.Writer) *baseWriter {
	return &baseWriter{w: w}
}

func (p *baseWriter) writeUint16(u uint16) error {
	buf := p.buf[:2]
	byteOrder.PutUint16(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *baseWriter) writeUint32(u uint32) error {
	buf := p.buf[:4]
	byteOrder.PutUint32(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *baseWriter) writeUint64(u uint64) error {
	buf := p.buf[:8]
	byteOrder.PutUint64(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *baseWriter) writeFloat32(v float32) error {
	u := math.Float32bits(v)
	return p.writeUint32(u)
}

func (p *baseWriter) writeFloat64(v float64) error {
	u := math.Float64bits(v)
	return p.writeUint64(u)
}

type baseReader struct {
	buf [8]byte
	r   io.Reader
}

func newBaseReader(r io.Reader) *baseReader {
	return &baseReader{r: r}
}

func (p *baseReader) readUint16() (uint16, error) {
	buf := p.buf[:2]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint16(buf), nil
}

func (p *baseReader) readUint32() (uint32, error) {
	buf := p.buf[:4]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint32(buf), nil
}

func (p *baseReader) readUint64() (uint64, error) {
	buf := p.buf[:8]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint64(buf), nil
}

func (p *baseReader) readFloat32() (float32, error) {
	v, err := p.readUint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

func (p *baseReader) readFloat64() (float64, error) {
	v, err := p.readUint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(v), nil
}
