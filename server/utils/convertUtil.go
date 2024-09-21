package utils

import (
	"encoding/json"
	"fmt"
)

/*
*
字符串转int数组
*/
func str2IntArr(arrStr string) any {
	var arr [][]int
	err := json.Unmarshal([]byte(arrStr), &arr)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	return arr
}

func intArr2ListNode(arr []int) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	for i := 0; i < len(arr); i++ {
		cur.Next = &ListNode{
			Val:  arr[i],
			Next: nil,
		}
		cur = cur.Next
	}
	return pivot.Next
}

// arrayToBST 将整数数组转换为二叉搜索树
func arrayToBST(nums []interface{}) *TreeNode {
	return buildTree(nums, 0, len(nums)-1)
}

// buildTree 是构建二叉搜索树的辅助函数
func buildTree(nums []interface{}, start, end int) *TreeNode {
	if start > end {
		return nil
	}
	mid := start + (end-start)/2
	if nums[mid] == nil {
		return buildTree(nums, start, mid-1) // 处理 nil 值
	}
	root := &TreeNode{Val: nums[mid].(int)}
	root.Left = buildTree(nums, start, mid-1)
	root.Right = buildTree(nums, mid+1, end)
	return root
}
