package utils

import (
	"crypto/md5"
	"encoding/hex"
	"math"
	sort "sort"
	"strings"
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

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
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

func (m MyImpl) RotateMatrix(matrix [][]int) {
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

func (m MyImpl) GetIntersectionNode(headA, headB *ListNode) *ListNode {
	A := headA
	B := headB
	for A != B {
		if A != nil {
			A = A.Next
		} else {
			A = headB
		}

		if B != nil {
			B = B.Next
		} else {
			B = headA
		}
	}
	return A
}

func (m MyImpl) SearchMatrix(matrix [][]int, target int) bool {
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

func (m MyImpl) ReverseList(head *ListNode) *ListNode {
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

// 空间O(1) 先从中间拆开，再反转后半段，相同则是回文链表
// 思路2，将链表拷贝到数组，然后双指针比较，空间O(n)
func (m MyImpl) IsPalindrome(head *ListNode) bool {
	if head.Next == nil {
		return true
	}
	var dfs func(cur, pre *ListNode) *ListNode
	dfs = func(cur, pre *ListNode) *ListNode {
		if cur == nil {
			return pre
		}
		res := dfs(cur.Next, cur)
		cur.Next = pre
		return res
	}
	mid := head
	fast := head
	for fast != nil && fast.Next != nil {
		mid = mid.Next
		fast = fast.Next.Next
	}
	cur := head
	for cur.Next != mid {
		cur = cur.Next
	}
	cur.Next = nil
	reverseList := dfs(mid, nil)
	A, B := head, reverseList
	for A.Next != nil && B.Next != nil {
		if A.Val != B.Val {
			return false
		}
		A = A.Next
		B = B.Next
	}
	if A.Val != B.Val {
		return false
	}
	return true
}

func (m MyImpl) hasCycle(head *ListNode) bool {
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if fast == slow {
			return true
		}
	}
	return false
}

func (m MyImpl) detectCycle(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast != nil {
		if fast.Next != nil {
			fast = fast.Next.Next
		} else {
			fast = nil
		}
		slow = slow.Next
		if fast == slow {
			break
		}
	}
	if fast == nil {
		return nil
	}
	fast = head
	for fast != slow {
		fast = fast.Next
		slow = slow.Next
	}
	return fast
}

func (m MyImpl) MergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	A, B := list1, list2
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	for A != nil && B != nil {
		if A.Val < B.Val {
			cur.Next = A
			A = A.Next
		} else {
			cur.Next = B
			B = B.Next
		}
		cur = cur.Next
		cur.Next = nil
	}
	if A != nil {
		cur.Next = A
	} else {
		cur.Next = B
	}
	return pivot.Next
}

func (m MyImpl) Md5Encode(str string) string {
	md5Ctx := md5.New()
	md5Ctx.Write([]byte(str))
	cipherStr := md5Ctx.Sum(nil)
	return hex.EncodeToString(cipherStr)
}

// 贪心
func (m MyImpl) FindContentChildren(g []int, s []int) int {
	k, n := len(g), len(s)
	res := 0
	sort.Ints(g)
	sort.Ints(s)
	for i, j := 0, 0; i < k; i++ {
		for j < n && s[j] < g[i] {
			j++
		}
		if j < n {
			res++
		}
		j++
	}
	return res
}

// 贪心，局部最优，每一个节点能跳的最远的位置，最后一个节点能跳的最远的位置理解为全局最优
func (m MyImpl) CanJump(nums []int) bool {
	k := 0
	for i := 0; i < len(nums); i++ {
		if i > k {
			return false
		}
		k = int(math.Max(float64(k), float64(i+nums[i])))
	}
	return true
}

// 贪心，局部最优，每一跳的小区间最优，最后全局最优
func (m MyImpl) Jump(nums []int) int {
	end, k := 0, 0
	res := 0
	for i := 0; i < len(nums); i++ {
		k = int(math.Max(float64(k), float64(i+nums[i])))
		if i == end {
			if end != len(nums)-1 {
				end = k
				res++
			}
		}
	}
	return res
}

func (m MyImpl) Fib(n int) int {
	if n < 2 {
		return n
	}
	x, y := 0, 1
	for i := 2; i <= n; i++ {
		tmp := y
		y, x = x+y, tmp
	}
	return y
}
func (m MyImpl) MinCostClimbingStairs(cost []int) int {
	dp := make([]int, len(cost)+1)
	dp[0], dp[1] = 0, 0
	for i := 2; i < len(dp); i++ {
		dp[i] = int(math.Min(float64(dp[i-2]+cost[i-2]), float64(dp[i-1]+cost[i-1])))
	}
	return dp[len(dp)-1]
}

func (m MyImpl) CanPartition(nums []int) bool {
	sum := 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
	}
	leftSum, rightSum := 0, 0
	for i := 0; i < len(nums); i++ {
		leftSum += nums[i]
		rightSum = sum - leftSum
		if leftSum == rightSum {
			return true
		}
	}
	return false
}
func (m MyImpl) Change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for _, coin := range coins {
		for i := coin; i < len(dp); i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[len(dp)-1]
}

func (m MyImpl) CountSubstrings(s string) int {
	dp := make([][]bool, len(s))
	res := 0
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	for j := 0; j < len(dp); j++ {
		for i := 0; i <= j; i++ {
			if s[i] == s[j] && (j-i < 2 || dp[i+1][j-1]) {
				dp[i][j] = true
				res++
			}
		}
	}
	return res
}

func (m MyImpl) LongestPalindromeSubseq(s string) int {
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	for i := len(dp) - 1; i >= 0; i-- {
		dp[i][i] = 1
		for j := i + 1; j < len(dp); j++ {
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = int(math.Max(float64(dp[i+1][j]), float64(dp[i][j-1])))
			}
		}
	}
	return dp[0][len(dp)-1]
}

func (m MyImpl) Combine(n int, k int) [][]int {
	res := make([][]int, 0)
	var dfs func(n, k int, ans []int)
	dfs = func(n, k int, ans []int) {
		if len(ans) >= k {
			tmp := make([]int, len(ans))
			copy(tmp, ans)
			res = append(res, tmp)
			return
		}
		for i := 1; i < n+1; i++ {
			if len(ans) >= 1 && ans[len(ans)-1] >= i {
				continue
			}
			ans = append(ans, i)
			dfs(n, k, ans)
			ans = ans[:len(ans)-1]
		}
	}
	dfs(n, k, make([]int, 0))
	return res
}
func (m MyImpl) CombinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	var dfs func(target int, ans []int, pre int)
	dfs = func(target int, ans []int, pre int) {
		if target <= 0 {
			if target == 0 {
				tmp := make([]int, len(ans))
				copy(tmp, ans)
				res = append(res, tmp)
			}
			return
		}
		for i := 0; i < len(candidates); i++ {
			if candidates[i] < pre {
				continue
			}
			ans = append(ans, candidates[i])
			dfs(target-candidates[i], ans, candidates[i])
			ans = ans[:len(ans)-1]
		}
	}
	sort.Ints(candidates)
	dfs(target, make([]int, 0), -1)
	return res
}

func (m MyImpl) CombinationSum2(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	var dfs func(target int, ans []int, start int)
	dfs = func(target int, ans []int, start int) {
		if target <= 0 {
			if target == 0 {
				tmp := make([]int, len(ans))
				copy(tmp, ans)
				res = append(res, tmp)
			}
		}
		for i := start; i < len(candidates); i++ {
			if target < candidates[i] {
				break
			}
			if i > start && candidates[i] == candidates[i-1] {
				continue
			}
			ans = append(ans, candidates[i])
			dfs(target-candidates[i], ans, i+1)
			ans = ans[:len(ans)-1]
		}
	}
	sort.Ints(candidates)
	dfs(target, make([]int, 0), 0)
	return res
}

//func (m MyImpl) SortedArrayToBST(nums []int) *TreeNode {
//	if len(nums) < 1 {
//		return nil
//	}
//	mid := len(nums) / 2
//	left := m.SortedArrayToBST(nums[:mid])
//	right := m.SortedArrayToBST(nums[mid+1:])
//	return &TreeNode{
//		Val:   nums[mid],
//		Left:  left,
//		Right: right,
//	}
//}

func (m MyImpl) SortList(head *ListNode) *ListNode {
	var dfs func(cur *ListNode, pre *ListNode) *ListNode
	dfs = func(cur *ListNode, pre *ListNode) *ListNode {
		if cur == nil {
			return pre
		}
		res := dfs(cur, cur.Next)
		cur.Next = pre
		return res
	}
	return dfs(head, nil)
}

func (m MyImpl) SolveNQueens(n int) [][]string {
	res := make([][]string, 0)
	var dfs func(ans [][]byte, depth int)
	check := func(ans [][]byte, row, column int) bool {
		for r := row - 1; r >= 0; {
			if ans[r][column] == 'Q' {
				return false
			}
			r--
		}
		for c := column - 1; c >= 0; {
			if ans[row][c] == 'Q' {
				return false
			}
			c--
		}
		for r, c := row-1, column-1; r >= 0 && c >= 0; {
			if ans[r][c] == 'Q' {
				return false
			}
			r--
			c--
		}
		for r, c := row-1, column+1; r >= 0 && c < len(ans[0]); {
			if ans[r][c] == 'Q' {
				return false
			}
			r--
			c++
		}
		return true
	}
	dfs = func(ans [][]byte, depth int) {
		if depth >= n {
			ans_tmp := make([]string, 0)
			for i := 0; i < len(ans); i++ {
				tmp := strings.Builder{}
				tmp.Write(ans[i])
				ans_tmp = append(ans_tmp, tmp.String())
			}
			res = append(res, ans_tmp)
			return
		}
		for j := 0; j < n; j++ {
			if !check(ans, depth, j) {
				continue
			}
			ans[depth][j] = 'Q'
			dfs(ans, depth+1)
			ans[depth][j] = '.'
		}
	}
	ans := make([][]byte, n)
	for i := 0; i < len(ans); i++ {
		ans[i] = make([]byte, 0)
		for j := 0; j < len(ans); j++ {
			ans[i] = append(ans[i], '.')
		}
	}
	dfs(ans, 0)
	return res
}

func (m MyImpl) sortList(head *ListNode) *ListNode {
	if head.Next == nil {
		return head
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	headT := slow.Next
	slow.Next = nil
	left := m.sortList(head)
	right := m.sortList(headT)
	A, B := left, right
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	for A != nil && B != nil {
		if A.Val < B.Val {
			cur.Next = A
			A = A.Next
		} else {
			cur.Next = B
			B = B.Next
		}
		cur = cur.Next
		cur.Next = nil
	}
	if A != nil {
		cur.Next = A
	} else {
		cur.Next = B
	}
	return pivot.Next
}

func (m MyImpl) MergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 1 {
		return lists[0]
	}
	if len(lists) == 0 {
		return nil
	}
	mid := len(lists) / 2
	left, right := lists[:mid], lists[mid:]
	A := m.MergeKLists(left)
	B := m.MergeKLists(right)
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	for A != nil && B != nil {
		if A.Val < B.Val {
			cur.Next = A
			A = A.Next
		} else {
			cur.Next = B
			B = B.Next
		}
		cur = cur.Next
		cur.Next = nil
	}
	if A != nil {
		cur.Next = A
	} else {
		cur.Next = B
	}
	return pivot.Next
}

func (m MyImpl) AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {

	A := l1
	B := l2
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	cur := pivot
	a, b := 0, 0
	carry, sum := 0, 0
	for A != nil || B != nil {
		if A != nil {
			a = A.Val
			A = A.Next
		} else {
			a = 0
		}
		if B != nil {
			b = B.Val
			B = B.Next
		} else {
			b = 0
		}
		sum = a + b + carry
		carry = sum / 10
		cur.Next = &ListNode{
			Val:  sum % 10,
			Next: nil,
		}
		cur = cur.Next
	}
	if carry != 0 {
		cur.Next = &ListNode{
			Val:  carry,
			Next: nil,
		}
	}
	r := pivot.Next
	return r
}

func (m MyImpl) RemoveNthFromEnd(head *ListNode, n int) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	A, B := pivot, pivot
	for i := 0; i < n; i++ {
		B = B.Next
	}
	for B != nil && B.Next != nil {
		A = A.Next
		B = B.Next
	}
	tmp := A.Next
	A.Next = tmp.Next
	tmp.Next = nil
	return pivot.Next
}

func (m MyImpl) SwapPairs(head *ListNode) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	cur := head
	pre := pivot
	for cur != nil && cur.Next != nil {
		tmp := cur.Next
		cur.Next = tmp.Next
		tmp.Next = cur
		pre.Next = tmp
		cur = cur.Next
		pre = pre.Next.Next
	}
	return pivot.Next
}

func (m MyImpl) InorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		res = append(res, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	return res
}

func (m MyImpl) MaxDepth(root *TreeNode) int {
	res := 0
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			res = int(math.Max(float64(res), float64(depth)))
			return
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 0)
	return res
}

func (m MyImpl) InvertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	l := m.InvertTree(root.Left)
	r := m.InvertTree(root.Right)
	root.Left = r
	root.Right = l
	return root
}
func (m MyImpl) IsSymmetric(root *TreeNode) bool {
	var dfs func(l, r *TreeNode) bool
	dfs = func(l, r *TreeNode) bool {
		if l == nil && r == nil {
			return true
		}
		if l == nil || r == nil || l.Val != r.Val {
			return false
		}
		return dfs(l.Left, r.Right) && dfs(l.Right, r.Left)
	}
	return dfs(root.Left, root.Right)
}

func (m MyImpl) DiameterOfBinaryTree(root *TreeNode) int {
	res := -1
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftDepth := dfs(node.Left)
		rightDepth := dfs(node.Right)
		res = int(math.Max(float64(res), float64(leftDepth+rightDepth)))
		return int(math.Max(float64(leftDepth), float64(rightDepth))) + 1
	}
	dfs(root)
	return res
}

func (m MyImpl) LevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	q := make([]*TreeNode, 0)
	q = append(q, root)
	res := make([][]int, 0)
	for len(q) != 0 {
		ans := make([]int, 0)
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			ans = append(ans, node.Val)
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
		res = append(res, ans)
	}
	return res
}

func (m MyImpl) SortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	midIdx := len(nums) / 2
	l := m.SortedArrayToBST(nums[:midIdx])
	r := m.SortedArrayToBST(nums[midIdx+1:])
	return &TreeNode{
		Val:   nums[midIdx],
		Left:  l,
		Right: r,
	}
}

func (m MyImpl) IsValidBST(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if root.Left != nil && root.Left.Val > root.Val {
		return false
	}
	if root.Right != nil && root.Right.Val < root.Val {
		return false
	}
	return m.IsValidBST(root.Left) && m.IsValidBST(root.Right)
}

func (m MyImpl) KthSmallest(root *TreeNode, k int) int {
	ans := make([]int, 0)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		ans = append(ans, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	return ans[k-1]
}
func (m MyImpl) RightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	q := make([]*TreeNode, 0)
	q = append(q, root)
	res := make([]int, 0)
	for len(q) != 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if i == size-1 {
				res = append(res, node.Val)
			}
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
	}
	return res
}

func (m MyImpl) flatten(root *TreeNode) {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		tmp := node.Right
		node.Right = dfs(node.Left)
		node.Left = nil
		r := dfs(tmp)
		cur := node
		for cur.Right != nil {
			cur = cur.Right
		}
		cur.Right = r
		return node
	}
	dfs(root)
}

func (m MyImpl) buildTree(preorder []int, inorder []int) *TreeNode {
	var dfs func(prearr []int, pre_l, pre_r int, inarr []int, in_l, in_r int) *TreeNode
	dfs = func(prearr []int, pre_l, pre_r int, inarr []int, in_l, in_r int) *TreeNode {
		if pre_l > pre_r {
			return nil
		}
		i := 0
		for prearr[pre_l] != inarr[i] {
			i++
		}
		i -= in_l
		l := dfs(prearr, pre_l+1, pre_l+i, inarr, in_l, in_l+i)
		r := dfs(prearr, pre_l+i+1, pre_r, inarr, in_l+i+1, in_r)
		return &TreeNode{
			Val:   prearr[pre_l],
			Left:  l,
			Right: r,
		}
	}
	return dfs(preorder, 0, len(preorder)-1, inorder, 0, len(inorder)-1)
}

func (m MyImpl) pathSum(root *TreeNode, targetSum int) int {
	preSumMap := make(map[int]int, 0)
	preSumMap[0] = 1
	ans := 0
	var dfs func(node *TreeNode, preSum int)
	dfs = func(node *TreeNode, preSum int) {
		if node == nil {
			return
		}
		preSum += node.Val
		ans += preSumMap[preSum-targetSum]
		preSumMap[preSum] += 1
		dfs(node.Left, preSum)
		dfs(node.Right, preSum)
		preSumMap[preSum] -= 1
	}
	dfs(root, 0)
	return ans
}

func (m MyImpl) lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	P := m.lowestCommonAncestor(root.Left, p, q)
	Q := m.lowestCommonAncestor(root.Right, p, q)
	if P != nil && Q != nil {
		return root
	}
	if P != nil {
		return P
	}
	if Q != nil {
		return Q
	}
	return nil
}

func (m MyImpl) MaxPathSum(root *TreeNode) int {
	res := 0
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		lm := int(math.Max(float64(m.MaxPathSum(node.Left)), float64(0)))
		rm := int(math.Max(float64(m.MaxPathSum(node.Right)), float64(0)))
		res = int(math.Max(float64(res), float64(node.Val+lm+rm)))
		return node.Val + int(math.Max(float64(lm), float64(rm)))
	}
	dfs(root)
	return res
}

func (m MyImpl) NumIslands(grid [][]byte) int {
	var dfs func(grid [][]byte, row, column int)
	dfs = func(grid [][]byte, row, column int) {
		if row < 0 || row > len(grid)-1 || column < 0 || column > len(grid[0])-1 || grid[row][column] == '0' {
			return
		}
		if grid[row][column] == '1' {
			grid[row][column] = '0'
		}
		dfs(grid, row-1, column)
		dfs(grid, row+1, column)
		dfs(grid, row, column-1)
		dfs(grid, row, column+1)
	}
	res := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				res++
				dfs(grid, i, j)
			}
		}
	}
	return res
}

func (m MyImpl) OrangesRotting(grid [][]int) int {
	type Node struct {
		X int
		Y int
	}
	q := make([]*Node, 0)
	cnt := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 2 {
				grid[i][j] = 0
				q = append(q, &Node{
					X: i,
					Y: j,
				})
			}
			if grid[i][j] == 1 {
				cnt++
			}
		}
	}
	var checkAndAdd func(x, y int)
	checkAndAdd = func(x, y int) {
		if x < 0 || x >= len(grid) || y < 0 || y >= len(grid[0]) || grid[x][y] == 0 {
			return
		}
		grid[x][y] = 0
		cnt--
		q = append(q, &Node{
			X: x,
			Y: y,
		})
	}
	res := 0
	for len(q) != 0 {
		size := len(q)
		res++
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			checkAndAdd(node.X+1, node.Y)
			checkAndAdd(node.X-1, node.Y)
			checkAndAdd(node.X, node.Y+1)
			checkAndAdd(node.X, node.Y-1)
		}
	}
	if cnt > 0 {
		return -1
	}
	if res == 0 {
		return res
	} else {
		return res - 1
	}
}

func (m MyImpl) CanFinish(numCourses int, prerequisites [][]int) bool {
	if len(prerequisites) == 0 {
		return true
	}
	type Node struct {
		Val    int
		Next   *Node
		degree int
	}
	nodeMap := make(map[int]*Node, 0)
	for i := 0; i < len(prerequisites); i++ {
		_, ok1 := nodeMap[prerequisites[i][0]]
		if !ok1 {
			nodeMap[prerequisites[i][0]] = &Node{
				Val:    prerequisites[i][0],
				Next:   nil,
				degree: 1,
			}
		} else {
			nodeMap[prerequisites[i][0]].degree++
		}
		_, ok := nodeMap[prerequisites[i][1]]
		if !ok {
			nodeMap[prerequisites[i][1]] = &Node{
				Val:    prerequisites[i][1],
				Next:   nodeMap[prerequisites[i][0]],
				degree: 0,
			}
		} else {
			nodeMap[prerequisites[i][1]].Next = nodeMap[prerequisites[i][0]]
		}
	}
	q := make([]*Node, 0)
	cnt := 0
	for _, node := range nodeMap {
		if node.degree == 0 {
			q = append(q, node)
			cnt++
		}
	}
	for len(q) != 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			tmp := q[0]
			q = q[1:]
			if tmp.Next == nil {
				continue
			}
			if tmp.Next.degree > 0 {
				tmp.Next.degree--
			}
			if tmp.Next.degree == 0 {
				q = append(q, tmp.Next)
				cnt++
			}
		}
	}
	if cnt >= numCourses || cnt == len(nodeMap) {
		return true
	} else {
		return false
	}
}

type Trie struct {
	root *TrieNode
}

type TrieNode struct {
	children [26]*TrieNode
	isEnd    bool
}

func Constructor() Trie {
	root := &TrieNode{
		children: [26]*TrieNode{},
		isEnd:    false,
	}
	return Trie{
		root: root,
	}
}

func (this *Trie) Insert(word string) {
	node := this.root
	for _, c := range word {
		if node.children[c-'a'] == nil {
			node.children[c-'a'] = &TrieNode{
				children: [26]*TrieNode{},
				isEnd:    false,
			}
		}
		node = node.children[c-'a']
	}
	node.isEnd = true
}

func (this *Trie) Search(word string) bool {
	node := this.root
	for _, c := range word {
		if node.children[c-'a'] == nil {
			return false
		} else {
			node = node.children[c-'a']
		}
	}
	if node.isEnd {
		return true
	}
	return false
}

func (this *Trie) StartsWith(prefix string) bool {
	node := this.root
	for _, c := range prefix {
		if node.children[c-'a'] == nil {
			return false
		} else {
			node = node.children[c-'a']
		}
	}
	return true
}

func (m MyImpl) permute(nums []int) [][]int {
	res := make([][]int, 0)
	var dfs func(nums []int, ans []int, visited []bool)
	dfs = func(nums []int, ans []int, visited []bool) {
		if len(ans) == len(nums) {
			tmp := make([]int, len(ans))
			copy(tmp, ans)
			res = append(res, tmp)
		}
		for i := 0; i < len(nums); i++ {
			if visited[i] {
				continue
			}
			ans = append(ans, nums[i])
			visited[i] = true
			dfs(nums, ans, visited)
			ans = ans[:len(ans)-1]
			visited[i] = false
		}
	}
	dfs(nums, make([]int, 0), make([]bool, len(nums)))
	return res
}

func (m MyImpl) subsets(nums []int) [][]int {
	res := make([][]int, 0)
	var dfs func(ans []int, start int)
	dfs = func(ans []int, start int) {
		tmp := make([]int, len(ans))
		copy(tmp, ans)
		res = append(res, tmp)
		for i := start; i < len(nums); i++ {
			ans = append(ans, nums[i])
			dfs(ans, i+1)
			ans = ans[:len(ans)-1]
		}
	}
	dfs(make([]int, 0), 0)
	return res
}

func (m MyImpl) letterCombinations(digits string) []string {
	digitMap := make(map[byte][]byte)
	digitMap['2'] = []byte{'a', 'b', 'c'}
	digitMap['3'] = []byte{'d', 'e', 'f'}
	digitMap['4'] = []byte{'g', 'h', 'i'}
	digitMap['5'] = []byte{'j', 'k', 'l'}
	digitMap['6'] = []byte{'m', 'n', 'o'}
	digitMap['7'] = []byte{'p', 'q', 'r', 's'}
	digitMap['8'] = []byte{'t', 'u', 'v'}
	digitMap['9'] = []byte{'w', 'x', 'y', 'z'}
	var dfs func(index int, ans []byte)
	res := make([]string, 0)
	dfs = func(index int, ans []byte) {
		if index >= len(digits) {
			res = append(res, string(ans))
			return
		}
		for i := 0; i < len(digitMap[digits[index]]); i++ {
			ans = append(ans, digitMap[digits[index]][i])
			dfs(index+1, ans)
			ans = ans[:len(ans)-1]
		}
	}
	if len(digits) == 0 {
		return []string{}
	}
	dfs(0, make([]byte, 0))
	return res
}
func (m MyImpl) combinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	var dfs func(ans []int, sum int, start int)
	dfs = func(ans []int, sum int, start int) {
		if sum < 0 {
			return
		}
		if sum == 0 {
			tmp := make([]int, len(ans))
			copy(tmp, ans)
			res = append(res, tmp)
			return
		}
		for i := start; i < len(candidates); i++ {
			ans = append(ans, candidates[i])
			dfs(ans, sum-candidates[i], i)
			ans = ans[:len(ans)-1]
		}
	}
	dfs(make([]int, 0), target, 0)
	return res
}
func (m MyImpl) partition(s string) [][]string {
	var check func(string) bool
	check = func(s string) bool {
		sb := strings.Builder{}
		for i := len(s) - 1; i >= 0; i-- {
			sb.WriteByte(s[i])
		}
		if sb.String() == s {
			return true
		} else {
			return false
		}
	}
	res := make([][]string, 0)
	var dfs func(ans []string, start, end int)
	dfs = func(ans []string, start, end int) {
		if start >= len(s) {
			tmp := make([]string, len(ans))
			copy(tmp, ans)
			res = append(res, tmp)
			return
		}
		for i := end + 1; i < len(s)+1; i++ {
			if check(s[start:i]) {
				ans = append(ans, s[start:i])
				dfs(ans, i, i)
				ans = ans[:len(ans)-1]
			}
		}
	}
	dfs(make([]string, 0), 0, 0)
	return res
}
func (m MyImpl) generateParenthesis(n int) []string {
	res := make([]string, 0)
	var dfs func(ans []byte, l, r int)
	dfs = func(ans []byte, l, r int) {
		if len(ans) >= 2*n {
			res = append(res, string(ans))
			return
		}
		if l < n {
			dfs(append(ans, '('), l+1, r)
		}
		if r < l && r < n {
			dfs(append(ans, ')'), l, r+1)
		}
	}
	dfs(make([]byte, 0), 0, 0)
	return res
}
func (m MyImpl) exist(board [][]byte, word string) bool {
	var dfs func(ans []byte, row, column int) bool
	dfs = func(ans []byte, row, column int) bool {
		if row < 0 || row >= len(board) || column < 0 || column >= len(board[0]) || board[row][column] == '0' {
			return false
		}
		ans = append(ans, board[row][column])
		if len(ans) >= len(word) {
			if string(ans) == word {
				return true
			}
			return false
		}
		tmp := board[row][column]
		board[row][column] = '0'
		res := dfs(ans, row+1, column) || dfs(ans, row-1, column) || dfs(ans, row, column+1) || dfs(ans, row, column-1)
		board[row][column] = tmp
		return res
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if dfs(make([]byte, 0), i, j) {
				return true
			}
		}
	}
	return false
}

func (m MyImpl) solveNQueens(n int) [][]string {
	type Node struct {
		X int
		Y int
	}
	res := make([][]string, 0)
	var check = func(r, c int, nodeList []*Node) bool {
		for _, node := range nodeList {
			diffX := node.X - r
			diffY := node.Y - c
			if diffX == 0 || diffY == 0 || math.Abs(float64(diffY)) == math.Abs(float64(diffX)) {
				return false
			}
		}
		return true
	}
	var dfs func(ans []string, row int, nodeList []*Node)
	dfs = func(ans []string, row int, nodeList []*Node) {
		if row >= n {
			tmp := make([]string, len(ans))
			copy(tmp, ans)
			res = append(res, tmp)
			return
		}
		for i := 0; i < n; i++ {
			if !check(row, i, nodeList) {
				continue
			}
			str := make([]byte, n)
			for j := 0; j < n; j++ {
				str[j] = '.'
			}
			str[i] = 'Q'
			ans = append(ans, string(str))
			nodeList = append(nodeList, &Node{
				X: row,
				Y: i,
			})
			dfs(ans, row+1, nodeList)
			ans = ans[:len(ans)-1]
			nodeList = nodeList[:len(nodeList)-1]
		}
	}
	dfs(make([]string, 0), 0, make([]*Node, 0))
	return res
}

func (m MyImpl) searchInsert(nums []int, target int) int {
	i, j := 0, len(nums)-1
	for i <= j {
		mid := (i + j) >> 1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			i = mid + 1
		} else {
			j = mid - 1
		}
	}
	return i
}

func (m MyImpl) searchMatrix(matrix [][]int, target int) bool {
	l, r := 0, len(matrix)-1
	row := -1
	for l <= r {
		mid := (l + r) >> 1
		if matrix[mid][0] == target {
			row = mid
			break
		} else if matrix[mid][0] < target {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	if row == -1 {
		row = r
	}
	if row == -1 {
		return false
	}
	l, r = 0, len(matrix[0])-1
	for l <= r {
		mid := (l + r) >> 1
		if matrix[row][mid] == target {
			return true
		} else if matrix[row][mid] < target {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return false
}
func (m MyImpl) search(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := (l + r) >> 1
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

func (m MyImpl) findMin(nums []int) int {
	l, r := 0, len(nums)-1
	min := math.MaxInt
	for l <= r {
		mid := (l + r) >> 1
		min = int(math.Min(float64(min), float64(nums[mid])))
		if nums[l] <= nums[mid] {
			min = int(math.Min(float64(min), float64(nums[l])))
			l = mid + 1
		} else {
			min = int(math.Min(float64(min), float64(nums[mid])))
			r = mid - 1
		}
	}
	return min
}

func (m MyImpl) findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	k := len(nums1) + len(nums2)
	res := 0.0
	var dfs func(start1, end1, start2, end2, m int)
	dfs = func(start1, end1, start2, end2, m int) {
		if m <= 0 {
			return
		}
		mid := m / 2
		if nums1[mid] <= nums2[mid] {
			if mid == 1 {
				res = float64(nums1[mid])
			}
			dfs(start1+mid, end1+mid, start2, end2, m-mid)
		} else {
			if mid == 1 {
				res = float64(nums2[mid])
			}
			dfs(start1, end1, start2+mid, end2+mid, m-mid)
		}
	}
	dfs(0, len(nums1)-1, 0, len(nums2)-1, k)
	return res
}
func (m MyImpl) isValid(s string) bool {
	q := make([]rune, 0)
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			q = append(q, c)
		} else {
			if len(q) == 0 {
				return false
			}
			top := q[len(q)-1]
			if (c == ')' && top == '(') || (c == ']' && top == '[') || (c == '}' && top == '{') {
				q = q[:len(q)-1]
			} else {
				return false
			}
		}
	}
	return len(q) == 0
}

func (m MyImpl) decodeString(s string) string {
	type Node struct {
		BeforeStr string
		Cnt       int
	}
	var res string
	q := make([]*Node, 0)
	count := 0
	for _, c := range s {
		if c == '[' {
			q = append(q, &Node{
				BeforeStr: res,
				Cnt:       count,
			})
			res = ""
			count = 0
		} else if '0' <= c && c <= '9' {
			count = count*10 + int(c-'0')
		} else if c == ']' {
			node := q[len(q)-1]
			q = q[:len(q)-1]
			tmp := node.BeforeStr
			for i := 0; i < node.Cnt; i++ {
				tmp += res
			}
			res = tmp
		} else {
			res += string(c)
		}
	}
	return res
}
func (m MyImpl) findKthLargest(nums []int, k int) int {
	var heapify func(nums []int, x, n int)
	heapify = func(nums []int, x, n int) {
		if x >= n {
			return
		}
		max := x
		left := (x << 1) + 1
		right := (x << 1) + 2
		if left < n && nums[left] > nums[max] {
			max = left
		}
		if right < n && nums[right] > nums[max] {
			max = right
		}
		if max != x {
			nums[x], nums[max] = nums[max], nums[x]
			heapify(nums, max, n)
		}
	}
	buildHeap := func(nums []int) {
		for i := (len(nums) - 2) >> 1; i >= 0; i-- {
			heapify(nums, i, len(nums))
		}
	}
	if len(nums) == 1 {
		return nums[0]
	}
	buildHeap(nums)
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[0]
	}
	return nums[len(nums)-k]
}

func (m MyImpl) dailyTemperatures(temperatures []int) []int {
	res := make([]int, len(temperatures))
	for i := len(temperatures) - 2; i >= 0; i-- {
		for j := i + 1; j < len(temperatures); j += res[j] {
			if temperatures[i] < temperatures[j] {
				res[i] = j - i
				break
			} else if res[j] == 0 {
				res[i] = 0
				break
			}
		}
	}
	return res
}

func (m MyImpl) topKFrequent(nums []int, k int) []int {
	cntMap := make(map[int]int, 0)
	for i := 0; i < len(nums); i++ {
		_, ok := cntMap[nums[i]]
		if ok {
			cntMap[nums[i]]++
		} else {
			cntMap[nums[i]] = 1
		}
	}
	type Entry struct {
		Val int
		Cnt int
	}
	var heapify func(arr []*Entry, x, n int)
	heapify = func(arr []*Entry, x, n int) {
		if x >= n {
			return
		}
		left := (x >> 1) + 1
		right := (x >> 1) + 2
		max := x
		if left < n && arr[left].Cnt < arr[max].Cnt {
			max = left
		}
		if right < n && arr[right].Cnt < arr[max].Cnt {
			max = right
		}
		if x != max {
			arr[x], arr[max] = arr[max], arr[x]
			heapify(arr, max, n)
		}
	}
	buildMinStack := func(arr []*Entry) {
		for i := (len(arr) >> 1) - 2; i >= 0; i-- {
			heapify(arr, i, len(arr))
		}
	}
	res := make([]*Entry, 0)
	count := 0
	for key, value := range cntMap {
		if count < k {
			res = append(res, &Entry{
				Val: key,
				Cnt: value,
			})
		} else {
			if count == k {
				buildMinStack(res)
			}
			if value > res[0].Cnt {
				res[0] = &Entry{
					Val: key,
					Cnt: value,
				}
				heapify(res, 0, len(res))
			}
		}
		count++
	}
	result := make([]int, 0)
	for i := 0; i < len(res); i++ {
		result = append(result, res[i].Val)
	}
	return result
}

func (m MyImpl) maxProfit(prices []int) int {
	min, res := prices[0], 0
	for i := 0; i < len(prices); i++ {
		res = int(math.Max(float64(res), float64(prices[i]-min)))
		min = int(math.Min(float64(min), float64(prices[i])))
	}
	return res
}

func (m MyImpl) canJump(nums []int) bool {
	l, r := 0, 0
	nextSkipIndex := 0
	for l <= r && r < len(nums)-1 {
		for i := l; i <= r; i++ {
			nextSkipIndex = int(math.Max(float64(nextSkipIndex), float64(i+nums[i])))
		}
		l = r + 1
		r = nextSkipIndex
	}
	return nextSkipIndex >= len(nums)-1
}
func (m MyImpl) jump(nums []int) int {
	l, r := 0, 0
	count := 0
	nextSkipIndex := 0
	for l <= r && r < len(nums)-1 {
		for i := l; i <= r; i++ {
			nextSkipIndex = int(math.Max(float64(nextSkipIndex), float64(i+nums[i])))
		}
		l = r + 1
		r = nextSkipIndex
		count++
	}
	return count
}

func (m MyImpl) partitionLabels(s string) []int {
	l, r := 0, 0
	a := make([]int, 26)
	res := make([]int, 0)
	for i := 0; i < len(s); i++ {
		a[s[i]-'a'] = int(math.Max(float64(a[s[i]-'a']), float64(i)))
	}
	for l <= r && r < len(s) {
		for i := l; i <= r; i++ {
			r = int(math.Max(float64(r), float64(a[s[i]-'a'])))
		}
		res = append(res, r-l+1)
		l, r = r+1, r+1
	}
	return res
}
func (m MyImpl) climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[len(dp)-1]
}
func (m MyImpl) generate(numRows int) [][]int {
	res := make([][]int, 0)
	for i := 0; i < numRows; i++ {
		row := make([]int, i+1)
		for j := 0; j < len(row); j++ {
			row[j] = 1
		}
		res = append(res, row)
	}
	for i := 2; i < numRows; i++ {
		for j := 1; j < len(res[i])-1; j++ {
			res[i][j] = res[i-1][j-1] + res[i-1][j]
		}
	}
	return res
}

func (m MyImpl) rob(nums []int) int {
	dp := make([]int, len(nums)+1)
	dp[0] = 0
	dp[1] = nums[0]
	for i := 2; i < len(dp); i++ {
		dp[i] = int(math.Max(float64(dp[i-1]), float64(dp[i-2]+nums[i-1])))
	}
	return dp[len(dp)-1]
}
func (m MyImpl) numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 1; i < len(dp); i++ {
		dp[i] = i
		for j := 1; i-j*j >= 0; j++ {
			dp[i] = int(math.Min(float64(dp[i]), float64(dp[i-j*j]+1)))
		}
	}
	return dp[len(dp)-1]
}
func (m MyImpl) coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 0; i < len(dp); i++ {
		for j := len(coins) - 1; i-coins[j] >= 0; j-- {
			dp[i] = int(math.Min(float64(dp[i]), float64(dp[i-coins[j]])))
		}
	}
	return dp[len(dp)-1]
}
func (m MyImpl) wordBreak(s string, wordDict []string) bool {
	dict := make(map[string]struct{})
	for _, c := range wordDict {
		dict[c] = struct{}{}
	}
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 0; i < len(s); i++ {
		for j := i + 1; j < len(dp); j++ {
			_, ok := dict[s[i:j]]
			if dp[i] && ok {
				dp[j] = dp[i] && ok
			}
			if dp[len(dp)-1] {
				return true
			}
		}
	}
	return false
}
func (m MyImpl) lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	res := 0
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
		for j := i - 1; j >= 0; j-- {
			if nums[j] < nums[i] {
				dp[i] = int(math.Max(float64(dp[i]), float64(dp[j]+1)))
			}
		}
		res = int(math.Max(float64(res), float64(dp[i])))
	}
	return res
}

func (m MyImpl) maxProduct(nums []int) int {
	dpMax := make([]int, len(nums))
	dpMin := make([]int, len(nums))
	dpMax[0], dpMin[0] = nums[0], nums[0]
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		dpMax[i] = int(math.Max(math.Max(float64(dpMax[i-1]*nums[i]), float64(dpMin[i-1]*nums[i])), float64(nums[i])))
		dpMin[i] = int(math.Min(math.Min(float64(dpMax[i-1]*nums[i]), float64(dpMin[i-1]*nums[i])), float64(nums[i])))
		res = int(math.Max(float64(dpMax[i]), float64(res)))
	}
	return res
}

func (m MyImpl) canPartition(nums []int) bool {
	pack := func(N, W int, arr []int) int {
		dp := make([][]int, N+1)
		for i := 0; i < len(dp); i++ {
			dp[i] = make([]int, W+1)
		}
		for i := 1; i <= N; i++ {
			for j := 1; j <= W; j++ {
				dp[i][j] = dp[i-1][j]
				if j >= nums[i-1] {
					dp[i][j] = int(math.Max(float64(dp[i-1][j]), float64(dp[i-1][j-nums[i-1]]+nums[i-1])))
				}
			}
		}
		return dp[N][W]
	}
	sum := 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
	}
	if sum%2 != 0 {
		return false
	}
	return pack(len(nums), sum>>1, nums) == sum>>1
}

func (m MyImpl) findTargetSumWays(nums []int, target int) int {
	sum := 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
	}
	diff := sum - target
	if diff < 0 || diff%2 != 0 {
		return 0
	}
	tar := diff >> 1
	dp := make([]int, tar+1)
	dp[0] = 1
	for i := 0; i < len(nums); i++ {
		for j := tar; j >= 0; j-- {
			if j >= nums[i] {
				dp[j] += dp[j-nums[i]]
			}
		}
	}
	return dp[len(dp)-1]
}

func (ml MyImpl) findMaxForm(strs []string, m int, n int) int {
	dp := make([][][]int, len(strs)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([][]int, m+1)
		for j := 0; j < len(dp[i]); j++ {
			dp[i][j] = make([]int, n+1)
		}
	}
	for i := 1; i < len(strs)+1; i++ {
		zeros, ones := 0, 0
		for _, c := range strs[i-1] {
			if c == '0' {
				zeros++
			} else {
				ones++
			}
		}
		for j := 0; j < m+1; j++ {
			for k := 0; k < n+1; k++ {
				dp[i][j][k] = dp[i-1][j][k]
				if j >= zeros && k >= ones && dp[i][j][k] < dp[i-1][j-zeros][k-ones]+1 {
					dp[i][j][k] = dp[i-1][j-zeros][k-ones] + 1
				}
			}
		}
	}
	return dp[len(strs)][m][n]
}
func (ml MyImpl) uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func (m MyImpl) minPathSum(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(grid[0]))
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < len(dp); i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for i := 1; i < len(dp[0]); i++ {
		dp[0][i] = dp[0][i-1] + grid[0][i]
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[0]); j++ {
			dp[i][j] = int(math.Min(float64(dp[i-1][j]), float64(dp[i][j-1]))) + grid[i][j]
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}
func (m MyImpl) longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	res := s[:1]
	dp[0][0] = true
	for j := 1; j < len(dp); j++ {
		for i := 0; i < j; i++ {
			if s[i] == s[j] {
				if j-i < 3 {
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
			} else {
				dp[i][j] = false
			}
			if dp[i][j] && j-i+1 > len(res) {
				res = s[i : j+1]
			}
		}
	}
	return res
}

func (m MyImpl) longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[0]); j++ {
			max := int(math.Max(math.Max(float64(dp[i-1][j-1]), float64(dp[i-1][j])), float64(dp[i][j-1])))
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

func (m MyImpl) minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for i := 1; i < len(dp[0]); i++ {
		dp[0][i] = i
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[0]); j++ {
			if word1[i-1] != word2[j-1] {
				dp[i][j] = int(math.Min(float64(dp[i-1][j-1]), math.Min(float64(dp[i-1][j]), float64(dp[i][j-1])))) + 1
			} else {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

func (m MyImpl) singleNumber(nums []int) int {
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		res ^= nums[i]
	}
	return res
}

func (m MyImpl) majorityElement(nums []int) int {
	res, cnt := nums[0], 1
	for i := 1; i < len(nums); i++ {
		if res != nums[i] {
			cnt--
			if cnt == 0 {
				res = nums[i]
				cnt = 1
			}
		} else {
			cnt++
		}
	}
	return res
}
