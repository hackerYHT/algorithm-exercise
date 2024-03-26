package utils

import (
	"math"
	sort "sort"
)

type Algorithm interface {
	groupAnagrams(strs []string) [][]string
	longestConsecutive(nums []int) int
}

type MyImpl struct {
	Algorithm
	Name string
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func (m MyImpl) GroupAnagrams(strs []string) [][]string {
	var sorts func(str string) string
	sorts = func(str string) string {
		s := []rune(str)
		sort.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		return string(s)
	}
	myMap := make(map[string][]string, 0)
	for _, str := range strs {
		s := sorts(str)
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

func (m MyImpl) LongestConsecutive(nums []int) int {
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
func (m MyImpl) MoveZeroes(nums []int) {
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

func (m MyImpl) MaxArea(height []int) int {
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

func (m MyImpl) ThreeSum(nums []int) [][]int {
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
				for l < r && nums[l] == nums[l+1] {
					l++
				}
				for l < r && nums[r] == nums[r-1] {
					r--
				}
				l++
				r--
			}
		}
	}
	return res
}

func (m MyImpl) Trap(height []int) int {
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

func (m MyImpl) LengthOfLongestSubstring(s string) int {
	if len(s) < 2 {
		return len(s)
	}
	i, j := 0, 0
	res := 0
	myMap := make(map[byte]int, 0)
	for j < len(s) {
		v, ok := myMap[s[j]]
		if ok && v >= i {
			i = myMap[s[j]] + 1
		}
		myMap[s[j]] = j
		res = int(math.Max(float64(res), float64(j-i+1)))
		j++
	}
	return res
}
func (m MyImpl) FindAnagrams(s string, p string) []int {
	if len(s) < len(p) {
		return nil
	}
	l, r := 0, len(p)-1
	res := make([]int, 0)
	var sArr [26]int
	var pArr [26]int
	for i, c := range p {
		pArr[c-'a'] += 1
		sArr[s[i]-'a'] += 1
	}
	for r < len(s) {
		if pArr == sArr {
			res = append(res, l)
		}
		if r < len(s) {
			sArr[s[l]-'a'] -= 1
		}
		l++
		r++
		if r < len(s) {
			sArr[s[r]-'a'] += 1
		}
	}
	return res
}

func (m MyImpl) SubarraySum(nums []int, k int) int {
	dp := make([]int, len(nums))
	res := 0
	if len(nums) == 0 {
		return 0
	}
	if nums[0] == k {
		dp[0] = 1
		res += 1
	} else {
		dp[0] = 0
	}
	for i := 1; i < len(dp); i++ {
		tmp := 0
		for j := i; j >= 0; j-- {
			tmp += nums[j]
			if tmp == k {
				dp[i]++
			}
		}
		res += dp[i]
	}
	return res
}

func (m MyImpl) MaxSubArray(nums []int) int {
	res := nums[0]
	cur, pre := 0, nums[0]
	for i := 1; i < len(nums); i++ {
		cur = int(math.Max(float64(pre+nums[i]), float64(nums[i])))
		res = int(math.Max(float64(res), float64(cur)))
		pre = cur
	}
	return res
}

func (m MyImpl) Merge(intervals [][]int) [][]int {
	res := make([][]int, 0)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	tmp := []int{-1, -1}
	for i := 0; i < len(intervals); i++ {
		if tmp[0] == -1 {
			tmp = intervals[i]
		} else {
			if tmp[1] >= intervals[i][0] {
				tmp[1] = int(math.Max(float64(intervals[i][1]), float64(tmp[1])))
			} else {
				res = append(res, tmp)
				tmp = intervals[i]
			}
		}
	}
	res = append(res, tmp)
	return res
}

func (m MyImpl) Rotate(nums []int, k int) {
	lenth := len(nums)
	tmp := make([]int, lenth)
	copy(tmp, nums)
	for i, t := range tmp {
		nums[(i+k)%lenth] = t
	}
}

func (m MyImpl) ProductExceptSelf(nums []int) []int {
	dpLeft := make([]int, len(nums))
	dpRight := make([]int, len(nums))
	dpLeft[0] = 1
	dpRight[len(nums)-1] = 1
	for i := 1; i < len(dpLeft); i++ {
		dpLeft[i] = dpLeft[i-1] * nums[i-1]
	}
	for i := len(dpRight) - 2; i >= 0; i-- {
		dpRight[i] = dpRight[i+1] * nums[i+1]
	}
	res := make([]int, len(nums))
	for i := 0; i < len(res); i++ {
		res[i] = dpLeft[i] * dpRight[i]
	}
	return res
}

func (m MyImpl) SetZeroes(matrix [][]int) {
	rowMap := make(map[int]int, 0)
	colmunMap := make(map[int]int, 0)
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if matrix[i][j] == 0 {
				rowMap[i] = 1
				colmunMap[j] = 1
			}
		}
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			_, ok1 := rowMap[i]
			_, ok2 := colmunMap[j]
			if ok1 || ok2 {
				matrix[i][j] = 0
			}
		}
	}
}

func (m MyImpl) SpiralOrder(matrix [][]int) []int {
	res := make([]int, 0)
	l, r, u, d := 0, len(matrix[0])-1, 0, len(matrix)-1
	for {
		for i := l; i <= r; i++ {
			res = append(res, matrix[u][i])
		}
		u++
		if u > d {
			break
		}
		for i := u; i <= d; i++ {
			res = append(res, matrix[i][r])
		}
		r--
		if l > r {
			break
		}
		for i := r; i >= l; i-- {
			res = append(res, matrix[d][i])
		}
		d--
		if u > d {
			break
		}
		for i := d; i >= u; i-- {
			res = append(res, matrix[i][l])
		}
		l++
		if l > r {
			break
		}
	}
	return res
}

func (m MyImpl) rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := 0; j < (n+1)/2; j++ {
			tmp := matrix[i][j]
			matrix[i][j] = matrix[n-1-j][i]
			matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
			matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
			matrix[j][n-1-i] = tmp
		}
	}
}

func (m MyImpl) getIntersectionNode(headA, headB *ListNode) *ListNode {
	A, B := headA, headB
	for A != nil && B != nil {
		if A == B {
			return A
		}
		A = A.Next
		B = B.Next
		if A == nil {
			A = headB
		}
		if B == nil {
			B = headA
		}
	}
	return nil
}

func (m MyImpl) searchMatrix(matrix [][]int, target int) bool {
	i, j := len(matrix)-1, 0
	for i >= 0 && j < len(matrix[0]) {
		for i >= 0 && j < len(matrix[0]) && matrix[i][j] > target {
			i--
		}
		for i >= 0 && j < len(matrix[0]) && matrix[i][j] < target {
			j++
		}
		if i >= 0 && j < len(matrix[0]) && matrix[i][j] == target {
			return true
		}
	}
	return false
}

func (m MyImpl) reverseList(head *ListNode) *ListNode {
	var dfs func(cur, pre *ListNode) *ListNode
	dfs = func(cur, pre *ListNode) *ListNode {
		if cur == nil {
			return pre
		}
		res := dfs(cur.Next, cur)
		cur.Next = pre
		return res
	}
	return dfs(head, nil)
}

func (m MyImpl) isPalindrome(head *ListNode) bool {
	var dfs func(cur, pre *ListNode) *ListNode
	dfs = func(cur, pre *ListNode) *ListNode {
		if cur == nil {
			return pre
		}
		res := dfs(cur.Next, cur)
		cur.Next = pre
		return res
	}
	reverseList := dfs(head, nil)
	A, B := head, reverseList
	for A != nil || B != nil {
		if A != B {
			return false
		}
		A = A.Next
		B = B.Next
	}
	return true
}

func (m MyImpl) detectCycle(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if fast == slow {
			return fast
		}
	}
	return nil
}

func (m MyImpl) mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	A, B := list1, list2
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	for A != nil || B != nil {
		if A.Val < B.Val {
			cur.Next = A
			cur = cur.Next
			cur.Next = nil
			A = A.Next
		}
	}
	if A != nil {
		cur.Next = A
	} else {
		cur.Next = B
	}
	return pivot.Next
}
