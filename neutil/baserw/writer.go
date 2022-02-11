package baserw

import (
	"io"
	"math"
)

type BaseWriter struct {
	buf [bytesPerUint64]byte
	w   io.Writer
}

func NewBaseWriter(w io.Writer) *BaseWriter {
	return &BaseWriter{w: w}
}

func (p *BaseWriter) WriteUint8(u uint8) error {
	buf := p.buf[:bytesPerUint8]
	buf[0] = u
	_, err := p.w.Write(buf)
	return err
}

func (p *BaseWriter) WriteUint16(u uint16) error {
	buf := p.buf[:bytesPerUint16]
	byteOrder.PutUint16(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *BaseWriter) WriteUint32(u uint32) error {
	buf := p.buf[:bytesPerUint32]
	byteOrder.PutUint32(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *BaseWriter) WriteUint64(u uint64) error {
	buf := p.buf[:bytesPerUint64]
	byteOrder.PutUint64(buf, u)
	_, err := p.w.Write(buf)
	return err
}

func (p *BaseWriter) WriteFloat32(v float32) error {
	u := math.Float32bits(v)
	return p.WriteUint32(u)
}

func (p *BaseWriter) WriteFloat64(v float64) error {
	u := math.Float64bits(v)
	return p.WriteUint64(u)
}
