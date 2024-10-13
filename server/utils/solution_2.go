package utils

import (
	"sort"
)

type MyImplTwo struct {
	Algorithm
	Name string
}

func (m MyImplTwo) findKthLargest(nums []int, k int) int {
	var heapify func(arr []int, index, lenth int)
	heapify = func(arr []int, index, lenth int) {
		if index >= lenth {
			return
		}
		left := index<<1 + 1
		right := index<<1 + 2
		max := index
		if left < lenth && arr[left] > arr[max] {
			max = left
		}
		if right < lenth && arr[right] > arr[max] {
			max = right
		}
		if max != index {
			arr[max], arr[index] = arr[index], arr[max]
			heapify(arr, max, lenth)
		}
	}
	for i := (len(nums) - 2) / 2; i >= 0; i-- {
		heapify(nums, i, len(nums))
	}
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[0]
	}
	return nums[len(nums)-k]
}

func threeSum(nums []int) [][]int {
	res := make([][]int, 0)
	if len(nums) < 3 {
		return res
	}
	if len(nums) == 3 && nums[0]+nums[1]+nums[2] == 0 {
		res = append(res, nums)
		return res
	}
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			break
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		target := 0 - nums[i]
		l, r := i+1, len(nums)-1
		for l < r {
			twoSum := nums[l] + nums[r]
			if twoSum < target {
				l++
			} else if twoSum > target {
				r--
			} else {
				res = append(res, []int{nums[i], nums[l], nums[r]})
				for ; l < r && nums[l] == nums[l+1]; l++ {
				}
				for ; l < r && nums[r] == nums[r-1]; r-- {
				}
				l++
				r--
			}
		}
	}
	return res
}

func (m MyImplTwo) numIslands(grid [][]byte) int {
	var dfs func(grid [][]byte, row, column int)
	dfs = func(grid [][]byte, row, column int) {
		if row < 0 || row >= len(grid) || column < 0 || column >= len(grid[row]) || grid[row][column] == '0' {
			return
		}
		grid[row][column] = '0'
		dfs(grid, row+1, column)
		dfs(grid, row, column+1)
		dfs(grid, row-1, column)
		dfs(grid, row, column-1)
	}
	res := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				res++
				dfs(grid, i, j)
			}
		}
	}
	return res
}
func (m MyImplTwo) reverseKGroup(head *ListNode, k int) *ListNode {
	var reverse func(cur, pre, end *ListNode) *ListNode
	reverse = func(cur, pre, end *ListNode) *ListNode {
		if cur == end || cur == nil {
			return pre
		}
		tmp := reverse(cur.Next, cur, end)
		cur.Next = pre
		return tmp
	}
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	start, end := pivot, pivot.Next
	for i := 0; i < k; i++ {
		end = end.Next
	}
	for {
		h := reverse(start.Next, end, end)
		start.Next = h
		for i := 0; i < k; i++ {
			if end == nil {
				return pivot.Next
			}
			start = start.Next
			end = end.Next
		}
	}
	return pivot.Next
}
func (m MyImplTwo) search(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := (l + r) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[l] <= nums[mid] {
			if nums[l] <= target && nums[mid] > target {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else {
			if nums[mid] < target && nums[r] >= target {
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return -1
}
