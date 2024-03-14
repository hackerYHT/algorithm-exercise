package lc

type Algorithm interface {
	groupAnagrams(strs []string) [][]string
	longestConsecutive(nums []int) int
}

type MyImpl struct {
	Algorithm
	Name string
}

func (m MyImpl) groupAnagrams(strs []string) [][]string {
	return nil
}

func (m MyImpl) longestConsecutive(nums []int) int {
	return 0
}
