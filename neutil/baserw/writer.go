package baserw

import (
	"encoding/binary"
	"io"
	"math"
)

type BaseWriter struct {
	w   io.Writer
	buf []byte
}

func NewBaseWriter(w io.Writer) *BaseWriter {
	return &BaseWriter{
		w:   w,
		buf: make([]byte, binary.MaxVarintLen64),
	}
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

//------------------------------------------------------------------------------

func (p *BaseWriter) WriteCompactUint64(u uint64) error {
	n := binary.PutUvarint(p.buf, u)
	_, err := p.w.Write(p.buf[:n])
	return err
}

func (p *BaseWriter) WriteCompactInt64(i int64) error {
	n := binary.PutVarint(p.buf, i)
	_, err := p.w.Write(p.buf[:n])
	return err
}

func (p *BaseWriter) WriteCompactInt(i int) error {
	return p.WriteCompactInt64(int64(i))
}

func (p *BaseWriter) WriteString(s string) error {
	var (
		bs = []byte(s)
		n  = len(bs)
	)
	err := p.WriteCompactInt(n)
	if err != nil {
		return err
	}
	_, err = p.w.Write(bs)
	return err
}
