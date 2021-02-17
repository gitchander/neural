package mnist

import (
	"path/filepath"
)

var (
	trainNames = SetFiles{
		Images: "train-images-idx3-ubyte.gz",
		Labels: "train-labels-idx1-ubyte.gz",
	}

	testNames = SetFiles{
		Images: "t10k-images-idx3-ubyte.gz",
		Labels: "t10k-labels-idx1-ubyte.gz",
	}
)

type SetFiles struct {
	Images string `json:"images"`
	Labels string `json:"labels"`
}

type DBFiles struct {
	TrainingSet SetFiles `json:"training-set"`
	TestSet     SetFiles `json:"test-set"`
}

func MakeDBFiles(directory string) DBFiles {
	return DBFiles{
		TrainingSet: SetFiles{
			Images: filepath.Join(directory, trainNames.Images),
			Labels: filepath.Join(directory, trainNames.Labels),
		},
		TestSet: SetFiles{
			Images: filepath.Join(directory, testNames.Images),
			Labels: filepath.Join(directory, testNames.Labels),
		},
	}
}
