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

func TestPartition(t *testing.T) {
	res := myimpl.partition("cbbbcc")
	fmt.Printf("res: %v", res)
}

func TestSolveNQueens(t *testing.T) {
	res := myimpl.solveNQueens(6)
	fmt.Printf("res: %v", res)
}

func TestSearchInsert(t *testing.T) {
	res := myimpl.searchInsert([]int{1, 3, 5, 6}, 7)
	fmt.Printf("res: %v", res)
}

func TestSearchMatrix(t *testing.T) {
	res := myimpl.searchMatrix([][]int{{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 60}}, 3)
	fmt.Printf("res: %v", res)
}

func TestSearch(t *testing.T) {
	res := myimpl.search([]int{4, 5, 6, 7, 0, 1, 2}, 3)
	fmt.Printf("res: %v", res)
}

func TestFindMin(t *testing.T) {
	res := myimpl.findMin([]int{11, 13, 15, 17})
	fmt.Printf("res: %v", res)
}
func TestDecodeString(t *testing.T) {
	res := myimpl.decodeString("3[a]2[bc]")
	fmt.Printf("res: %v", res)
}
