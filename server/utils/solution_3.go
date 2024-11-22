package utils

import "math"

type MyImplThree struct {
	Algorithm
	Name string
}

func (m MyImplThree) lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	res := 0
	for i := 1; i < len(dp); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[j] <= nums[i] {
				dp[i] = int(math.Max(float64(dp[i]), float64(dp[j]+1)))
			}
		}
		res = int(math.Max(float64(dp[i]), float64(res)))
	}
	return res
}
