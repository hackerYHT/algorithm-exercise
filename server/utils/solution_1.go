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

func (m MyImpl) longestValidParentheses(s string) int {
	return 1
}

func (m MyImplOne) minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for i := 0; i < len(dp[0]); i++ {
		dp[0][i] = i
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[0]); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = int(math.Min(math.Min(float64(dp[i-1][j]), float64(dp[i][j-1])), float64(dp[i-1][j-1]))) + 1
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}
