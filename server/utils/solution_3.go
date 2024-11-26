package utils

import "math"

type MyImplThree struct {
	Algorithm
	Name string
}

func (m MyImplThree) lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	res := 1
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	for i := 1; i < len(dp); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[j] < nums[i] {
				dp[i] = int(math.Max(float64(dp[i]), float64(dp[j]+1)))
			}
		}
		res = int(math.Max(float64(dp[i]), float64(res)))
	}
	return res
}
func (m MyImplThree) getIntersectionNode(headA, headB *ListNode) *ListNode {
	A, B := headA, headB
	for A != B {
		if A == nil {
			A = headB
		}
		if B == nil {
			B = headA
		}
		A = A.Next
		B = B.Next
	}
	return A
}
func (im MyImplThree) numWays(n, m int) int {
	dp := make([][]int, n)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, m)
		dp[i][0] = 1
	}
	for i := 0; i < len(dp[0]); i++ {
		dp[0][i] = 1
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[0]); j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[n-1][m-1]
}

func (m MyImplThree) findKthLargest(nums []int, k int) int {
	var heapify func(arr []int, x, len int)
	heapify = func(arr []int, x, len int) {
		if x > len {
			return
		}
		l, r := x*2+1, x*2+2
		max := x
		if l < len && arr[l] > arr[max] {
			max = l
		}
		if r < len && arr[r] > arr[max] {
			max = r
		}
		if max != x {
			arr[max], arr[x] = arr[x], arr[max]
			heapify(arr, max, len)
		}
	}
	buildHeap := func(arr []int) {
		for i := (len(arr) - 2) / 2; i >= 0; i-- {
			heapify(arr, i, len(arr)-1)
		}
	}
	buildHeap(nums)
	res := 0
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i] = nums[len(nums)-i], nums[0]
	}
	return res
}
