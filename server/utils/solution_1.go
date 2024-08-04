package utils

import "math"

type MyImplOne struct {
	Algorithm
	Name string
}

func (m MyImplOne) findKthLargest(nums []int, k int) int {
	var heapify func(arr []int, x, n int)
	heapify = func(arr []int, x, n int) {
		if x >= n {
			return
		}
		l, r, max := x<<1+1, x<<1+2, x
		if l < n && nums[l] > nums[max] {
			max = l
		}
		if r < n && nums[r] > nums[max] {
			max = r
		}
		if max != x {
			nums[x], nums[max] = nums[max], nums[x]
			heapify(arr, max, n)
		}
	}
	buildHeap := func(arr []int) {
		for i := (len(nums) - 2) >> 1; i >= 0; i-- {
			heapify(arr, i, len(nums))
		}
	}
	buildHeap(nums)
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[0]
	}
	return nums[len(nums)-k]
}

func (m MyImplOne) maxProfit(prices []int) int {
	res, min := 0, prices[0]
	for i := 0; i < len(prices); i++ {
		min = int(math.Min(float64(min), float64(prices[i])))
		res = int(math.Max(float64(prices[i]-min), float64(res)))
	}
	return res
}
