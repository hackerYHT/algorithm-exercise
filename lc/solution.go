package lc

import (
	"math"
	sort2 "sort"
)

type Algorithm interface {
	groupAnagrams(strs []string) [][]string
	longestConsecutive(nums []int) int
}

type MyImpl struct {
	Algorithm
	Name string
}

func (m MyImpl) groupAnagrams(strs []string) [][]string {
	var sort func(str string) string
	sort = func(str string) string {
		s := []rune(str)
		sort2.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		return string(s)
	}
	myMap := make(map[string][]string, 0)
	for _, str := range strs {
		s := sort(str)
		_, ok := myMap[s]
		if ok {
			myMap[s] = append(myMap[s], str)
		} else {
			myMap[s] = append(myMap[s], []string{str}...)
		}
	}
	res := make([][]string, 0)
	for _, strings := range myMap {
		ans := make([]string, 0)
		for _, s := range strings {
			ans = append(ans, s)
		}
		res = append(res, ans)
	}
	return res
}

func (m MyImpl) longestConsecutive(nums []int) int {
	myMap := make(map[int]int, 0)
	res := 0
	for _, num := range nums {
		_, ok := myMap[num]
		if !ok {
			left, ok := myMap[num-1]
			if !ok {
				left = 0
			}
			right, ok := myMap[num+1]
			if !ok {
				right = 0
			}
			tmp := left + right + 1
			myMap[num] = tmp
			myMap[num-left] = tmp
			myMap[num+right] = tmp
			res = int(math.Max(float64(res), float64(tmp)))
		}
	}
	return res
}
func (m MyImpl) moveZeroes(nums []int) {
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			for j := i + 1; j < len(nums); j++ {
				if nums[j] != 0 {
					nums[i], nums[j] = nums[j], nums[i]
					break
				}
			}
		}
	}
}

func (m MyImpl) maxArea(height []int) int {
	i, j := 0, len(height)-1
	res := 0
	for i < j {
		a, b := 0, j-i
		if height[i] < height[j] {
			a = height[i]
			i++
		} else {
			a = height[j]
			j--
		}
		ans := a * b
		res = int(math.Max(float64(res), float64(ans)))
	}
	return res
}

func (m MyImpl) threeSum(nums []int) [][]int {
	res := make([][]int, 0)
	if len(nums) < 3 {
		return res
	}
	sort2.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			break
		}
		target := 0 - nums[i]
		l, r := i+1, len(nums)-1
		for l < r {
			twoSum := nums[l] + nums[r]
			if l < r && twoSum < target {
				l++
			} else if l < r && twoSum > target {
				r--
			} else {
				res = append(res, []int{nums[i], nums[l], nums[r]})
			}
			for l < r && nums[l] == nums[l+1] {
				l++
			}
			for l < r && nums[r] == nums[r-1] {
				r--
			}
		}
	}
	return res
}

func (m MyImpl) trap(height []int) int {
	res := 0
	dpLeft := make([]int, len(height))
	dpRight := make([]int, len(height))
	dpLeft[0], dpRight[len(dpRight)-1] = 0, 0
	for i := 1; i < len(dpLeft); i++ {
		dpLeft[i] = int(math.Max(float64(dpLeft[i-1]), float64(height[i-1])))
	}
	for i := len(dpRight) - 2; i >= 0; i-- {
		dpRight[i] = int(math.Max(float64(dpRight[i+1]), float64(height[i+1])))
	}
	for i := 0; i < len(height); i++ {
		tmp := int(math.Min(float64(dpLeft[i]), float64(dpRight[i])))
		if tmp > height[i] {
			res += tmp - height[i]
		}
	}
	return res
}
