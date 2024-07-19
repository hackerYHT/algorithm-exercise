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

func TestDailyTemperatures(t *testing.T) {
	res := myimpl.dailyTemperatures([]int{73, 74, 75, 71, 69, 72, 76, 73})
	fmt.Printf("res: %v", res)
}
func TestDecodeString(t *testing.T) {
	res := myimpl.decodeString("3[a]2[bc]")
	fmt.Printf("res: %v", res)
}
func TestFindKthLargest(t *testing.T) {
	res := myimpl.findKthLargest([]int{3, 2, 1, 5, 6, 4}, 2)
	fmt.Printf("res: %v", res)
}
func TestTopKFrequent(t *testing.T) {
	res := myimpl.topKFrequent([]int{5, 3, 1, 1, 1, 3, 73, 1}, 2)
	fmt.Printf("res: %v", res)
}

func TestClimbStairs(t *testing.T) {
	res := myimpl.climbStairs(5)
	fmt.Printf("res: %v", res)
}

func TestWordBreak(t *testing.T) {
	res := myimpl.wordBreak("applepenapple", []string{"apple", "pen"})
	fmt.Printf("res: %v", res)
}

func TestLengthOfLISk(t *testing.T) {
	res := myimpl.lengthOfLIS([]int{1, 3, 6, 7, 9, 4, 10, 5, 6})
	fmt.Printf("res: %v", res)
}

func TestMaxProduct(t *testing.T) {
	res := myimpl.maxProduct([]int{-2, 3, -4})
	fmt.Printf("res: %v", res)
}

func TestCanPartition(t *testing.T) {
	res := myimpl.canPartition([]int{1, 5, 11, 5})
	fmt.Printf("res: %#v", res)
}

func TestFindTargetSumWays(t *testing.T) {
	res := myimpl.findTargetSumWays([]int{1, 1, 1, 1, 1}, 3)
	fmt.Printf("res: %+v", res)
}

func TestFindMaxForm(t *testing.T) {
	res := myimpl.findMaxForm([]string{"10", "0001", "111001", "1", "0"}, 5, 3)
	fmt.Printf("res: %+v", res)
}

func TestUniquePaths(t *testing.T) {
	res := myimpl.uniquePaths(3, 2)
	fmt.Printf("res: %+v", res)
}

func TestMinPathSum(t *testing.T) {
	res := myimpl.minPathSum([][]int{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}})
	fmt.Printf("res: %+v", res)
}

func TestMinDistance(t *testing.T) {
	res := myimpl.minDistance("horse", "ros")
	fmt.Printf("res: %+v", res)
}

func TestSingleNumber(t *testing.T) {
	res := myimpl.singleNumber([]int{1, 6, 6, 8, 8, 1, 7, 9, 9, 3, 3})
	fmt.Printf("res: %+v", res)
}

func TestMajorityElement(t *testing.T) {
	res := myimpl.majorityElement([]int{2, 2, 1, 1, 1, 2, 2})
	fmt.Printf("res: %+v", res)
}
