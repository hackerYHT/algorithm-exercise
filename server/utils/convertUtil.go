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
