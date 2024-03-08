package baserw

// import (
// 	"bytes"
// 	"encoding/binary"
// 	"io"
// )

// func _() {
// 	r := bytes.NewReader(nil)
// 	r.ReadByte()
// }

// func WriteCompactInt64(w io.Writer, size int64) error {
// 	var (
// 		data = make([]byte, binary.MaxVarintLen64)
// 		n    = binary.PutVarint(data, size)
// 	)
// 	_, err := w.Write(data[:n])
// 	return err
// }

// func ReadCompactInt64(r io.Reader) (int64, error) {

// 	panic("todo")

// 	return 0, nil
// }
