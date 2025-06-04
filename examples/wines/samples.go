package main

import (
	"os"

	gone "github.com/gitchander/neural/goneural"
	"github.com/gocarina/gocsv"
)

func makeSamplesFile(filename string) (samples []gone.Sample, err error) {
	ws, err := readWines(filename)
	if err != nil {
		return nil, err
	}
	samples = make([]gone.Sample, len(ws))
	for i, w := range ws {
		samples[i] = makeSample(w)
	}
	return samples, nil
}

type WineInfo struct {
	WineClass           int     `csv:"wine_class"`
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
	Proline             float64 `csv:"proline"`
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

func makeSample(w *WineInfo) gone.Sample {
	return gone.Sample{
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
			w.Proline,
		},
		Outputs: gone.OneHot(3, w.WineClass-1),
	}
}
