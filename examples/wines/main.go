package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gitchander/neural"
	"github.com/gitchander/neural/neutil"
	"github.com/gocarina/gocsv"
)

func main() {
	ws, err := readWines("wine.csv")
	checkError(err)
	samples := make([]neural.Sample, len(ws))
	for i, w := range ws {
		samples[i] = makeSample(w)
	}
	neutil.NormalizeInputs(samples)
	p, err := neural.NewMLP(13, 3, 3)
	checkError(err)
	p.RandomizeWeights(neutil.NewRand())
	bp := neural.NewBP(p)
	bp.SetLearningRate(0.6)
	const epsilon = 0.001
	epochMax := 1000
	for epoch := 0; epoch < epochMax; epoch++ {
		averageCost, err := bp.LearnSamples(samples)
		checkError(err)
		if averageCost < epsilon {
			fmt.Println("Success!")
			fmt.Printf("average cost: %.7f\n", averageCost)
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

type WineInfo struct {
	WineType            int     `csv:"wine_type"`
	Alcohol             float64 `csv:"alcohol"`
	MalicAcid           float64 `csv:"malic_acid"`
	Ash                 float64 `csv:"ash"`
	Alcalinity          float64 `csv:"alcalinity"`
	Magnesium           float64 `csv:"magnesium"`
	Phenols             float64 `csv:"phenols"`
	Flavanoids          float64 `csv:"flavanoids"`
	NonflavanoidPhenols float64 `csv:"nonflavanoid_phenols"`
	Proanth             float64 `csv:"proanthocyanins"`
	ColorIntensity      float64 `csv:"color_intensity"`
	Hue                 float64 `csv:"hue"`
	OD                  float64 `csv:"od"`
	Proline             int     `csv:"proline"`
}

func readWines(filename string) (ws []*WineInfo, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	err = gocsv.UnmarshalFile(file, &ws)
	return ws, err
}

func makeSample(w *WineInfo) neural.Sample {
	return neural.Sample{
		Inputs: []float64{
			w.Alcohol,
			w.MalicAcid,
			w.Ash,
			w.Alcalinity,
			w.Magnesium,
			w.Phenols,
			w.Flavanoids,
			w.NonflavanoidPhenols,
			w.Proanth,
			w.ColorIntensity,
			w.Hue,
			w.OD,
			float64(w.Proline),
		},
		Outputs: neutil.OneHot(3, w.WineType-1),
	}
}
