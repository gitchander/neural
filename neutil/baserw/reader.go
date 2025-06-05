package baserw

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"
)

type BaseReader struct {
	br  *bufio.Reader
	buf []byte
}

func NewBaseReader(r io.Reader) *BaseReader {
	return &BaseReader{
		br:  bufio.NewReader(r),
		buf: make([]byte, binary.MaxVarintLen64),
	}
}

func (p *BaseReader) readFull(data []byte) error {
	_, err := io.ReadFull(p.br, data)
	return err
}

func (p *BaseReader) ReadUint8() (uint8, error) {
	buf := p.buf[:bytesPerUint8]
	err := p.readFull(buf)
	if err != nil {
		return 0, err
	}
	return buf[0], nil
}

func (p *BaseReader) ReadUint16() (uint16, error) {
	buf := p.buf[:bytesPerUint16]
	err := p.readFull(buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint16(buf), nil
}

func (p *BaseReader) ReadUint32() (uint32, error) {
	buf := p.buf[:bytesPerUint32]
	err := p.readFull(buf)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint32(buf), nil
}

func (p *BaseReader) ReadUint64() (uint64, error) {
	buf := p.buf[:bytesPerUint64]
	err := p.readFull(buf)
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

func (p *BaseReader) ReadFloat64s() ([]float64, error) {
	n, err := p.ReadCompactInt()
	if err != nil {
		return nil, err
	}
	vs := make([]float64, n)
	for i := range vs {
		v, err := p.ReadFloat64()
		if err != nil {
			return nil, err
		}
		vs[i] = v
	}
	return vs, nil
}

//------------------------------------------------------------------------------

func (p *BaseReader) ReadCompactUint64() (uint64, error) {
	return binary.ReadUvarint(p.br)
}

func (p *BaseReader) ReadCompactInt64() (int64, error) {
	return binary.ReadVarint(p.br)
}

func (p *BaseReader) ReadCompactInt() (int, error) {
	i, err := p.ReadCompactInt64()
	if err != nil {
		return 0, err
	}
	return int64ToInt(i)
}

func (p *BaseReader) ReadString() (string, error) {
	n, err := p.ReadCompactInt()
	if err != nil {
		return "", err
	}
	data := make([]byte, n)
	err = p.readFull(data)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
