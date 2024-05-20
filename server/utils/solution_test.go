package utils

import (
	"fmt"
	"testing"
)

var myimpl = MyImpl{
	Algorithm: nil,
	Name:      "test",
}

func TestPermute(t *testing.T) {
	res := myimpl.permute([]int{1, 2, 3})
	fmt.Printf("res: %v", res)
}
