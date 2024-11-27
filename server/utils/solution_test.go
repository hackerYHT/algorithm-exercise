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

var myimplOne = MyImplOne{
	Algorithm: nil,
	Name:      "test",
}

var myimplyTwo = MyImplTwo{
	Algorithm: nil,
	Name:      "test",
}

var myimplyThree = MyImplThree{
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

func TestSortColors(t *testing.T) {
	myimpl.sortColors([]int{2, 2, 1, 1, 0, 0, 0, 1, 2, 2})
}

func TestNextPermutation(t *testing.T) {
	res := []int{1, 2, 3}
	myimpl.nextPermutation(res)
	fmt.Printf("res: %+v", res)
}

func TestFindDuplicate(t *testing.T) {
	res := myimpl.findDuplicate([]int{1, 3, 4, 2, 2})
	fmt.Printf("res: %+v", res)
}

func TestLengthOfLongestSubstring(t *testing.T) {
	res := myimpl.lengthOfLongestSubstring("abba")
	fmt.Printf("res: %+v", res)
}

func TestSpiralOrder(t *testing.T) {
	res := myimpl.spiralOrder([][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})
	fmt.Printf("res: %+v", res)
}
func TestFindKthLargest_1(t *testing.T) {
	res := myimplOne.findKthLargest([]int{3, 2, 1, 5, 6, 4}, 2)
	fmt.Printf("res: %v", res)
}

// 将整数数组转换为单向链表
func (n ListNode) ArrayToLinkedList(nums []int) *ListNode {
	if len(nums) == 0 {
		return nil
	}

	dummy := &ListNode{} // 创建一个哑节点作为链表的头节点
	prev := dummy

	for _, val := range nums {
		prev.Next = &ListNode{Val: val}
		prev = prev.Next
	}

	return dummy.Next // 返回链表的第一个节点
}

func TestReverseBetween(t *testing.T) {
	node := &ListNode{
		Val:  -1,
		Next: nil,
	}
	head := node.ArrayToLinkedList([]int{1, 2, 3, 4, 5})
	res := myimpl.reverseBetween(head, 2, 4)
	//head := node.ArrayToLinkedList([]int{5})
	//res := myimpl.reverseBetween(head, 1, 1)
	for res != nil {
		fmt.Printf("%v ", res.Val)
		res = res.Next
	}
}
func TestPermute1(t *testing.T) {
	res := myimplOne.permute([]int{1, 2, 3})
	fmt.Printf("res: %v", res)
}

func TestMaxSlidingWindow(t *testing.T) {
	res := myimpl.maxSlidingWindow([]int{1, 3, -1, -3, 5, 3, 6, 7}, 3)
	fmt.Printf("res: %v", res)
}

func TestFindWords(t *testing.T) {
	res := myimpl.findWords([][]byte{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}}, []string{"oath", "pea", "eat", "rain"})
	fmt.Printf("res: %v", res)
}

func TestCombinationSum1(t *testing.T) {
	res := myimplOne.combinationSum([]int{2, 3, 6, 7}, 7)
	fmt.Printf("res: %v", res)
}

func TestSortList(t *testing.T) {
	res := myimplOne.sortList(intArr2ListNode([]int{4, 2, 1, 3}))
	fmt.Printf("res: \n")
	for res != nil {
		fmt.Printf("%v\n", res.Val)
		res = res.Next
	}
}
func TestMyAtoi(t *testing.T) {
	res := myimpl.myAtoi("9223372036854775808")
	fmt.Printf("res: %v", res)
}

func TestWidthOfBinaryTree(t *testing.T) {
	root := arrayToBST([]interface{}{1, 3, 2, 5, 3, nil, 9})
	res := myimplOne.widthOfBinaryTree(root)
	fmt.Printf("res: %v", res)
}

func TestReverseKGroup(t *testing.T) {
	head := intArr2ListNode([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	res := myimplyTwo.reverseKGroup(head, 2)
	for res != nil {
		fmt.Printf("%v\n", res.Val)
		res = res.Next
	}
}

func TestTrap(t *testing.T) {
	res := myimplyTwo.trap([]int{4, 2, 0, 3, 2, 5})
	fmt.Printf("%v\n", res)
}

func TestLongestPalindrome(t *testing.T) {
	res := myimplyTwo.longestPalindrome("babad")
	fmt.Printf("%v\n", res)
}

func TestNextPermutation2(t *testing.T) {
	myimplyTwo.nextPermutation([]int{1, 3, 2})
}

func TestMerge_1(t *testing.T) {
	res := myimplyTwo.merge_1([][]int{{1, 4}, {1, 5}})
	fmt.Printf("%v\n", res)
}
func TestMaxiMalSquare(t *testing.T) {
	res := myimplyTwo.maxiMalSquare([][]byte{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}, {'1', '0', '0', '1', '0'}})
	fmt.Printf("%v\n", res)
}
func TestSortList_1(t *testing.T) {
	res := myimplyTwo.sortList(intArr2ListNode([]int{4, 2, 1, 3}))
	fmt.Printf("%v\n", res)
}
func TestMysqrt(t *testing.T) {
	res := myimplyTwo.mySqrt(8)
	fmt.Printf("%v\n", res)
}
func TestCompareVersion(t *testing.T) {
	res := myimplyTwo.compareVersion("1.2", "1.10")
	fmt.Printf("%v\n", res)
}
func TestDeleteNode(t *testing.T) {
	root := arrayToBST([]interface{}{nil, 1, 2})
	res := myimplyThree.deleteNode(root, 1)
	fmt.Printf("res: %v", res)
}
