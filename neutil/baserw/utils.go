package baserw

import (
	"fmt"
	"math"
)

func int64ToInt(a int64) (int, error) {
	if a < math.MinInt {
		return 0, fmt.Errorf("int64 to int: (value = %d) < (math.MinInt = %d)", a, math.MinInt)
	}
	if a > math.MaxInt {
		return 0, fmt.Errorf("int64 to int: (value = %d) > (math.MaxInt = %d)", a, math.MaxInt)
	}
	return int(a), nil
}
