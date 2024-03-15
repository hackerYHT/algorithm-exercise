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
