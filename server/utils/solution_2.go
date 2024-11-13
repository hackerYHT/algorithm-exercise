package utils

import (
	"math"
	"sort"
)

type MyImplTwo struct {
	Algorithm
	Name string
}

func (m MyImplTwo) findKthLargest(nums []int, k int) int {
	var heapify func(arr []int, index, lenth int)
	heapify = func(arr []int, index, lenth int) {
		if index >= lenth {
			return
		}
		left := index<<1 + 1
		right := index<<1 + 2
		max := index
		if left < lenth && arr[left] > arr[max] {
			max = left
		}
		if right < lenth && arr[right] > arr[max] {
			max = right
		}
		if max != index {
			arr[max], arr[index] = arr[index], arr[max]
			heapify(arr, max, lenth)
		}
	}
	for i := (len(nums) - 2) / 2; i >= 0; i-- {
		heapify(nums, i, len(nums))
	}
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[0]
	}
	return nums[len(nums)-k]
}

func threeSum(nums []int) [][]int {
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
				for ; l < r && nums[l] == nums[l+1]; l++ {
				}
				for ; l < r && nums[r] == nums[r-1]; r-- {
				}
				l++
				r--
			}
		}
	}
	return res
}

func (m MyImplTwo) numIslands(grid [][]byte) int {
	var dfs func(grid [][]byte, row, column int)
	dfs = func(grid [][]byte, row, column int) {
		if row < 0 || row >= len(grid) || column < 0 || column >= len(grid[row]) || grid[row][column] == '0' {
			return
		}
		grid[row][column] = '0'
		dfs(grid, row+1, column)
		dfs(grid, row, column+1)
		dfs(grid, row-1, column)
		dfs(grid, row, column-1)
	}
	res := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				res++
				dfs(grid, i, j)
			}
		}
	}
	return res
}
func (m MyImplTwo) reverseKGroup(head *ListNode, k int) *ListNode {
	var reverse func(cur, pre, end *ListNode) *ListNode
	reverse = func(cur, pre, end *ListNode) *ListNode {
		if cur == end || cur == nil {
			return pre
		}
		tmp := reverse(cur.Next, cur, end)
		cur.Next = pre
		return tmp
	}
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	start, end := pivot, pivot.Next
	for i := 0; i < k; i++ {
		end = end.Next
	}
	for {
		h := reverse(start.Next, end, end)
		start.Next = h
		for i := 0; i < k; i++ {
			if end == nil {
				return pivot.Next
			}
			start = start.Next
			end = end.Next
		}
	}
	return pivot.Next
}
func (m MyImplTwo) search(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := (l + r) / 2
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

func (m MyImplTwo) trap(height []int) int {
	if len(height) < 3 {
		return 0
	}
	dp_l := make([]int, len(height))
	res := 0
	dp_l[0], dp_l[1] = 0, height[0]
	for i := 2; i < len(dp_l); i++ {
		dp_l[i] = int(math.Max(float64(dp_l[i-1]), float64(height[i-1])))
	}
	dp_r := make([]int, len(height))
	dp_r[len(dp_r)-1], dp_r[len(dp_r)-2] = 0, height[len(height)-1]
	for i := len(height) - 3; i >= 0; i-- {
		dp_r[i] = int(math.Max(float64(dp_r[i+1]), float64(height[i+1])))
	}
	for i := 1; i < len(height); i++ {
		min := int(math.Min(float64(dp_r[i]), float64(dp_l[i])))
		if dp_l[i] > height[i] && dp_r[i] > height[i] {
			res += min - height[i]
		}
	}
	return res
}
func (m MyImplTwo) spiralOrder(matrix [][]int) []int {
	res := make([]int, 0)
	l, r, t, b := 0, len(matrix[0])-1, 0, len(matrix)-1
	for l <= r && t <= b {
		for i := l; i <= r; i++ {
			res = append(res, matrix[t][i])
		}
		t++
		if t > b {
			break
		}
		for i := t; i <= b; i++ {
			res = append(res, matrix[i][r])
		}
		r--
		if l > r {
			break
		}
		for i := r; i >= l; i-- {
			res = append(res, matrix[b][i])
		}
		b--
		if t > b {
			break
		}
		for i := b; i >= t; i-- {
			res = append(res, matrix[i][l])
		}
		l++
		if l > r {
			break
		}
	}
	return res
}
func (m MyImplTwo) longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	res, max := "", -1
	for j := 0; j < len(s); j++ {
		for i := j; i >= 0; i-- {
			if j-i < 2 && s[i] == s[j] {
				dp[i][j] = true
			}
			if i+1 < len(s) && j-1 >= 0 && dp[i+1][j-1] && s[i] == s[j] {
				dp[i][j] = true
			}
			if dp[i][j] && j-i+1 > max {
				max = j - i + 1
				res = s[i : j+1]
			}
		}
	}
	return res
}
func (m MyImplTwo) lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		if node == p || node == q {
			return node
		}
		l := dfs(node.Left)
		r := dfs(node.Right)
		if l != nil && r != nil {
			return node
		}
		if l != nil {
			return l
		} else {
			return r
		}
	}
	return dfs(root)
}

func (m MyImplTwo) getIntersectionNode(headA, headB *ListNode) *ListNode {
	A, B := headA, headB
	for A != nil || B != nil {
		if A == nil {
			A = headB
		}
		if B == nil {
			B = headA
		}
		if A == B {
			return A
		}
		A = A.Next
		B = B.Next
	}
	return nil
}
func (m MyImplTwo) maxSubArray(nums []int) int {
	dp := make([]int, len(nums)+1)
	dp[0] = 0
	res := math.MinInt
	for i := 1; i < len(dp); i++ {
		dp[i] = int(math.Max(float64(dp[i-1]+nums[i-1]), float64(nums[i-1])))
		if res < dp[i] {
			res = dp[i]
		}
	}
	return res
}
func (m MyImplTwo) nextPermutation(nums []int) {
	sortArr := func(arr []int, l, r int) {
		for l <= r {
			nums[l], nums[r] = nums[r], nums[l]
			l++
			r--
		}
	}
	for i := len(nums) - 2; i >= 0; i-- {
		if nums[i] < nums[i+1] {
			j := len(nums) - 1
			for nums[j] <= nums[i] {
				j--
			}
			nums[i], nums[j] = nums[j], nums[i]
			sortArr(nums, i+1, len(nums)-1)
			return
		}
	}
	sort.Ints(nums)
}

func (m MyImplTwo) lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	res := 1
	dp[0] = 1
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	for i := 1; i < len(dp); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] && dp[i] < dp[j]+1 {
				dp[i] = dp[j] + 1
			}
		}
		if dp[i] > res {
			res = dp[i]
		}
	}
	return res
}
func (m MyImplTwo) rightSideView(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	q := make([]*TreeNode, 0)
	q = append(q, root)
	res := make([]int, 0)
	for len(q) != 0 {
		size := len(q)
		val := -1
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
			val = node.Val
		}
		res = append(res, val)
	}
	return res
}

func (m MyImplTwo) reorderList(head *ListNode) {
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	headA := head
	headB := slow.Next
	slow.Next = nil
	var reverse func(pre, cur *ListNode) *ListNode
	reverse = func(pre, cur *ListNode) *ListNode {
		if cur == nil {
			return pre
		}
		tmp := reverse(cur, cur.Next)
		cur.Next = pre
		return tmp
	}
	headB = reverse(nil, headB)
	A, B := headA, headB
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	flag := true
	cur := pivot
	for A != nil && B != nil {
		if flag {
			cur.Next = A
			A = A.Next
		} else {
			cur.Next = B
			B = B.Next
		}
		cur = cur.Next
		flag = !flag
	}
	if A != nil {
		cur.Next = A
	}
	if B != nil {
		cur.Next = B
	}
}

func (m MyImplTwo) isValid(s string) bool {
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, ')')
		} else if s[i] == '{' {
			stack = append(stack, '}')
		} else if s[i] == '[' {
			stack = append(stack, ']')
		} else {
			if len(stack) == 0 {
				return false
			}
			c := stack[len(stack)-1]
			if c != s[i] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	if len(stack) != 0 {
		return false
	} else {
		return true
	}
}

func (im MyImplTwo) merge(nums1 []int, m int, nums2 []int, n int) {
	a, b := m-1, n-1
	idx := len(nums1) - 1
	for b >= 0 {
		if a >= 0 && nums1[a] >= nums2[b] {
			nums1[idx] = nums1[a]
			a--
		} else {
			nums1[idx] = nums2[b]
			b--
		}
		idx--
	}
}

func (m MyImplTwo) reverseBetween(head *ListNode, left int, right int) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	pre, cur, tar := pivot, pivot.Next, pivot.Next
	for i := 0; i < right; i++ {
		if i < left-1 {
			pre = pre.Next
			cur = cur.Next
		}
		tar = tar.Next
	}
	for cur.Next != tar {
		tmp := cur.Next
		cur.Next = tmp.Next
		tmp.Next = pre.Next
		pre.Next = tmp
	}
	return pivot.Next
}

func (m MyImplTwo) mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: nil,
	}
	A, B, cur := list1, list2, pivot
	for A != nil && B != nil {
		if A.Val <= B.Val {
			cur.Next = A
			A = A.Next
		} else {
			cur.Next = B
			B = B.Next
		}
		cur = cur.Next
	}
	if A != nil {
		cur.Next = A
	} else {
		cur.Next = B
	}
	return pivot.Next
}

func (m MyImplTwo) hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		if fast == slow {
			return true
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return false
}

func (m MyImplTwo) minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for i := 0; i < len(dp[0]); i++ {
		dp[0][i] = i
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if word1[i-1] != word2[j-1] {
				dp[i][j] = int(math.Min(float64(dp[i-1][j-1]), math.Min(float64(dp[i-1][j]), float64(dp[i][j-1])))) + 1
			} else {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}
func (m MyImplTwo) merge_1(intervals [][]int) [][]int {
	res := make([][]int, 0)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	ans := make([]int, 0)
	for i := 0; i < len(intervals); i++ {
		if len(ans) == 0 {
			ans = append(ans, intervals[i][0])
			ans = append(ans, intervals[i][1])
		}
		if ans[1] >= intervals[i][0] {
			ans[1] = int(math.Max(float64(intervals[i][1]), float64(ans[1])))
		} else {
			tmp := make([]int, 2)
			copy(tmp, ans)
			res = append(res, tmp)
			ans[0], ans[1] = intervals[i][0], intervals[i][1]
		}
		if i == len(intervals)-1 {
			res = append(res, ans)
		}
	}
	return res
}

func (m MyImplTwo) maxiMalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	res := 0
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(matrix[i]))
		dp[i][0] = int(matrix[i][0] - '0')
		res = int(math.Max(float64(res), float64(dp[i][0])))
	}
	for i := 0; i < len(dp[0]); i++ {
		dp[0][i] = int(matrix[0][i] - '0')
		res = int(math.Max(float64(res), float64(dp[0][i])))
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if matrix[i][j] == '0' {
				dp[i][j] = 0
			} else {
				dp[i][j] = int(math.Min(float64(dp[i-1][j-1]), math.Min(float64(dp[i-1][j]), float64(dp[i][j-1])))) + 1
			}
			res = int(math.Max(float64(res), float64(dp[i][j])))
		}
	}
	return res * res
}

func (m MyImplTwo) sortList(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		pivot := &ListNode{
			Val:  -1,
			Next: node,
		}
		slow, fast := pivot, pivot.Next
		for fast != nil && fast.Next != nil {
			slow = slow.Next
			fast = fast.Next.Next
		}
		mid := slow.Next
		slow.Next = nil
		h1 := dfs(pivot.Next)
		h2 := dfs(mid)
		pivot.Next = nil
		cur := pivot
		for h1 != nil && h2 != nil {
			if h1.Val < h2.Val {
				cur.Next = h1
				h1 = h1.Next
			} else {
				cur.Next = h2
				h2 = h2.Next
			}
			cur = cur.Next
		}
		if h1 != nil {
			cur.Next = h1
		} else {
			cur.Next = h2
		}
		return pivot.Next
	}
	return dfs(head)
}
func (m MyImplTwo) mySqrt(x int) int {
	l, r := 0, x/2
	for l <= r {
		mid := (l + r) / 2
		if mid*mid < x {
			l = mid + 1
		} else if mid*mid > x {
			r = mid - 1
		} else {
			return mid
		}
	}
	return l - 1
}
