package neutil

import (
	"math/rand"
	"time"
)

func NewRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
}
