package neural

import (
	"bufio"
	"os"
)

func WriteFile(filename string, p *MLP) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	bw := bufio.NewWriter(file)
	defer bw.Flush()
	return Encode(bw, p)
}

func ReadFile(filename string) (*MLP, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	br := bufio.NewReader(file)
	return Decode(br)
}
