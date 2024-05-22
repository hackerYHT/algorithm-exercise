package utils

import (
	"fmt"
	"reflect"
	"testing"
)

var myimpl = MyImpl{
	Algorithm: nil,
	Name:      "test",
}

func TestPermute(t *testing.T) {
	str := "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]"
	res := myimpl.permute([]int{1, 2, 3})
	ok := reflect.DeepEqual(res, str2IntArr(str))
	if !ok {
		t.Fatal("test permute failed")
	}
}

func TestSubset(t *testing.T) {
	res := myimpl.subsets([]int{1, 2, 3})
	fmt.Printf("res: %v", res)
}
func TestLetterCombinations(t *testing.T) {
	res := myimpl.letterCombinations("23")
	fmt.Printf("res: %v", res)
}
func TestCombinationSum(t *testing.T) {
	res := myimpl.combinationSum([]int{2, 3, 6, 7}, 7)
	fmt.Printf("res: %v", res)
}
