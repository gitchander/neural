package main

import (
	"errors"
	"log"
	"os"

	"github.com/gitchander/neural/dataset/mnist/mnist_png"
)

func main() {
	checkError(drawDataset())
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func drawDataset() error {

	if len(os.Args) < 2 {
		return errors.New("There is no argument")
	}

	dirname := os.Args[1]

	{
		samples, err := mnist_png.ReadTraining(dirname)
		if err != nil {
			return err
		}

		err = mnist_png.MakeImages("mnist_png/train", samples)
		if err != nil {
			return err
		}
	}

	{
		samples, err := mnist_png.ReadTesting(dirname)
		if err != nil {
			return err
		}

		err = mnist_png.MakeImages("mnist_png/test", samples)
		if err != nil {
			return err
		}
	}

	return nil
}

// func saveImagePNG(im image.Image, filename string) error {
// 	var buf bytes.Buffer
// 	err := png.Encode(&buf, im)
// 	if err != nil {
// 		return err
// 	}
// 	return ioutil.WriteFile(filename, buf.Bytes(), 0664)
// }
