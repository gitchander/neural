package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gitchander/neural/dataset/mnist_png"
)

func main() {
	checkError(run())
}

func checkError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func run() error {

	if len(os.Args) < 2 {
		return fmt.Errorf("there are no arguments")
	}

	dirname := os.Args[1]

	ds, err := mnist_png.ReadDataSet(dirname)
	if err != nil {
		return err
	}

	fmt.Printf("%s set has %d samples\n", "Training", len(ds.Training))
	fmt.Printf("%s set has %d samples\n", "Testing", len(ds.Testing))

	return nil
}
