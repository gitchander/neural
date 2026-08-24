package mnist_gz

var byteToFloat = func() (table [256]float64) {
	for i := range table {
		table[i] = float64(i) / 255
	}
	return
}()
