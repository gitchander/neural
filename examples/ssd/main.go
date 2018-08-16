package main

import (
	"bytes"
	"fmt"
	"log"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
)

func main() {
	exampleSSD()
}

// seven-segment display (SSD)
func exampleSSD() {

	var digits = []uint{
		0x0: 0x3F,
		0x1: 0x06,
		0x2: 0x5B,
		0x3: 0x4F,
		0x4: 0x66,
		0x5: 0x6D,
		0x6: 0x7D,
		0x7: 0x07,
		0x8: 0x7F,
		0x9: 0x6F,
		0xA: 0x77,
		0xB: 0x7C,
		0xC: 0x39,
		0xD: 0x5E,
		0xE: 0x79,
		0xF: 0x71,
	}

	samples := make([]neural.Sample, len(digits))

	for i, d := range digits {
		samples[i] = neural.Sample{
			Inputs:  bitsToFloats(uint(i), 4),
			Outputs: bitsToFloats(d, 7),
		}
	}

	//	for _, sample := range samples {
	//		fmt.Println(PrintableSSD(sample.Outputs, "\t"))
	//	}
	//	return

	p, err := neural.NewMLP(4, 20, 7)
	checkError(err)
	p.RandomizeWeights(neutil.NewRand())
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.7)

	const epsilon = 0.001
	const epochMax = 100000
	for epoch := 0; epoch < epochMax; epoch++ {
		le, err := bp.LearnSamples(samples)
		checkError(err)
		if le < epsilon {
			fmt.Println("Success!")
			fmt.Printf("error: %.7f\n", le)
			fmt.Println("epoch =", epoch)
			return
		}
	}
	fmt.Println("Failure")
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func boolToFloat(b bool) float64 {
	if b {
		return 0.9
	}
	return 0.1
}

func floatToBool(f float64) bool {
	return f > 0.5
}

func bitsToFloats(x uint, n int) []float64 {
	vs := make([]float64, n)
	for i := range vs {
		vs[i] = boolToFloat((x & 1) == 1)
		x >>= 1
	}
	return vs
}

// -0000-
// 5----1
// 5----1
// 5----1
// -6666-
// 4----2
// 4----2
// 4----2
// -3333-

func PrintableSSD(vs []float64, prefix string) string {
	const (
		b_fill = '-'
		b_off  = '+'
		b_on   = '0'
	)
	ssb := make([][]byte, 9)
	for i := range ssb {
		sb := make([]byte, 6)
		for j := range sb {
			sb[j] = b_fill
		}
		ssb[i] = sb
	}
	for i, v := range vs {
		var b byte = b_off
		if floatToBool(v) {
			b = b_on
		}
		switch i {
		case 0:
			ssb[0][1] = b
			ssb[0][2] = b
			ssb[0][3] = b
			ssb[0][4] = b
		case 1:
			ssb[1][5] = b
			ssb[2][5] = b
			ssb[3][5] = b
		case 2:
			ssb[5][5] = b
			ssb[6][5] = b
			ssb[7][5] = b
		case 3:
			ssb[8][1] = b
			ssb[8][2] = b
			ssb[8][3] = b
			ssb[8][4] = b
		case 4:
			ssb[5][0] = b
			ssb[6][0] = b
			ssb[7][0] = b
		case 5:
			ssb[1][0] = b
			ssb[2][0] = b
			ssb[3][0] = b
		case 6:
			ssb[4][1] = b
			ssb[4][2] = b
			ssb[4][3] = b
			ssb[4][4] = b
		}
	}

	var buf bytes.Buffer
	for _, bs := range ssb {
		buf.WriteString(prefix)
		buf.Write(bs)
		buf.WriteByte('\n')
	}
	return buf.String()
}
