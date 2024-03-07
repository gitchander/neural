package random

import (
	"crypto/rand"
	"encoding/binary"
)

var NextSeed = func() func() int64 {
	seedsChan := make(chan int64)
	go func() {
		const BytesPerUint64 = 8
		var (
			n    = 100
			data = make([]byte, (n * BytesPerUint64))
		)
		for {
			_, err := rand.Read(data)
			if err != nil {
				panic(err)
			}
			for i := 0; i < n; i++ {
				bs := data[(i * BytesPerUint64):]
				u := binary.BigEndian.Uint64(bs)
				u = (u << 1) >> 1 // clear sign bit
				seedsChan <- int64(u)
			}
		}
	}()
	return func() int64 {
		return <-seedsChan
	}
}()
