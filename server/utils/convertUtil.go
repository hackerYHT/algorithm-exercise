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
