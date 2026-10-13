package goneural

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/gitchander/neural/neutil/random"
)

type interval struct {
	min, max float64
}

func makeInterval(min, max float64) interval {
	return interval{min, max}
}

func (v interval) clamp(x float64) float64 {
	if x < v.min {
		x = v.min
	}
	if x > v.max {
		x = v.max
	}
	return x
}

func (v interval) normalize(x float64) float64 {
	return (x - v.min) / (v.max - v.min)
}

func randWeight(r *rand.Rand) float64 {
	return random.FloatByInterval(r, -0.5, 0.5)
}

func clampFloat64(x float64, min, max float64) float64 {
	if x < min {
		x = min
	}
	if x > max {
		x = max
	}
	return x
}

//------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/One-hot

// Examples:
// OneHot(3, 0): [1, 0, 0]
// OneHot(3, 1): [0, 1, 0]
// OneHot(3, 2): [0, 0, 1]
// OneHot(4, 0): [1, 0, 0, 0]
// OneHot(4, 3): [0, 0, 0, 1]

func OneHot(n, i int) []float64 {
	vs := make([]float64, n)
	if n > 0 {
		vs[i] = 1
	}
	return vs
}

//------------------------------------------------------------------------------

func IndexOfMin(as []float64) int {
	n := len(as)
	if n == 0 {
		panic("IndexOfMin: slice is empty")
	}
	imin := 0
	for i := 1; i < n; i++ {
		if as[imin] > as[i] {
			imin = i
		}
	}
	return imin
}

func IndexOfMax(as []float64) int {
	n := len(as)
	if n == 0 {
		panic("IndexOfMax: slice is empty")
	}
	imax := 0
	for i := 1; i < n; i++ {
		if as[imax] < as[i] {
			imax = i
		}
	}
	return imax
}

//------------------------------------------------------------------------------

func NormalizeInputs(samples []Sample) {

	n := len(samples)
	if n == 0 {
		return
	}

	var vs = samples[0].Inputs
	var rs = make([]interval, len(vs))
	for i, v := range vs {
		rs[i] = makeInterval(v, v)
	}

	for k := 1; k < n; k++ {
		var vs = samples[k].Inputs
		for i, v := range vs {
			if v < rs[i].min {
				rs[i].min = v
			}
			if v > rs[i].max {
				rs[i].max = v
			}
		}
	}

	for _, sample := range samples {
		var vs = sample.Inputs
		for i, v := range vs {
			vs[i] = rs[i].normalize(v)
		}
	}
}

//------------------------------------------------------------------------------

type syncMap struct {
	guard sync.Mutex
	m     map[string]any
}

func newSyncMap() *syncMap {
	return &syncMap{
		m: make(map[string]any),
	}
}

func (p *syncMap) Set(key string, value any) {

	p.guard.Lock()
	defer p.guard.Unlock()

	p.m[key] = value
}

func (p *syncMap) Get(key string) (value any, ok bool) {

	p.guard.Lock()
	defer p.guard.Unlock()

	v, ok := p.m[key]
	return v, ok
}

//------------------------------------------------------------------------------

func sumFloat64s(as []float64) float64 {
	var sum float64
	for _, a := range as {
		sum += a
	}
	return sum
}

// almostEqual
func float64Equal(a, b float64, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func checkFloat64IsReal(v float64) error {
	if math.IsNaN(v) {
		return fmt.Errorf("value is NaN")
	}
	if math.IsInf(v, -1) {
		return fmt.Errorf("value is -Inf")
	}
	if math.IsInf(v, +1) {
		return fmt.Errorf("value is +Inf")
	}
	return nil
}

const logEpsilon = 1e-15

func safeLog(x float64) float64 {
	if x < logEpsilon {
		x = logEpsilon
	}
	y := math.Log(x)
	err := checkFloat64IsReal(y)
	if err != nil {
		panic(err)
	}
	return y
}
