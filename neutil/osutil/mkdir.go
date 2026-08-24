package osutil

import (
	"fmt"
	"os"
)

func MkdirIfNotExist(dirname string) error {
	fi, err := os.Stat(dirname)
	if err != nil {
		if os.IsNotExist(err) {
			return os.Mkdir(dirname, os.ModePerm)
		}
		return err
	}
	if !(fi.IsDir()) {
		return fmt.Errorf("name %q is not a directory", dirname)
	}
	return nil
}
