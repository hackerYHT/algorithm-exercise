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
