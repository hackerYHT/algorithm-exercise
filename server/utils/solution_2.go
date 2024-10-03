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
