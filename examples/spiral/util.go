package main

import (
	"log"
	"math/rand"
	"time"
)

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func newRandNow() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
}
