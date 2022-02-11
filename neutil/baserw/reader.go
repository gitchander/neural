package baserw

import (
	"io"
	"math"
)

type BaseReader struct {
	buf [bytesPerUint64]byte
	r   io.Reader
}

func NewBaseReader(r io.Reader) *BaseReader {
	return &BaseReader{r: r}
}

func (p *BaseReader) ReadUint8() (uint8, error) {
	buf := p.buf[:bytesPerUint8]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return buf[0], nil
}

func (p *BaseReader) ReadUint16() (uint16, error) {
	buf := p.buf[:bytesPerUint16]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint16(buf), nil
}

func (p *BaseReader) ReadUint32() (uint32, error) {
	buf := p.buf[:bytesPerUint32]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint32(buf), nil
}

func (p *BaseReader) ReadUint64() (uint64, error) {
	buf := p.buf[:bytesPerUint64]
	_, err := io.ReadFull(p.r, buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint64(buf), nil
}

func (p *BaseReader) ReadFloat32() (float32, error) {
	v, err := p.ReadUint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

func (p *BaseReader) ReadFloat64() (float64, error) {
	v, err := p.ReadUint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(v), nil
}
