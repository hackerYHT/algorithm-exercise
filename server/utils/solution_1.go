package utils

import (
	"math"
	"sort"
)

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
func (m MyImplOne) permute(nums []int) [][]int {
	res := make([][]int, 0)
	var dfs func(arr, ans []int, visited []bool)
	dfs = func(arr, ans []int, visited []bool) {
		if len(ans) == len(arr) {
			tmp := make([]int, len(arr))
			copy(tmp, ans)
			res = append(res, tmp)
		}
		for i := 0; i < len(arr); i++ {
			if visited[i] {
				continue
			}
			ans = append(ans, arr[i])
			visited[i] = true
			dfs(arr, ans, visited)
			ans = ans[:len(ans)-1]
			visited[i] = false
		}
	}
	dfs(nums, make([]int, 0), make([]bool, len(nums)))
	return res
}

func (m MyImplOne) findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	getKthElement := func(nums1, nums2 []int, k int) int {
		i1, i2 := 0, 0 // 当前开始下标
		for {
			//边界
			if i1 == len(nums1) {
				return nums2[i2+k-1]
			}
			if i2 == len(nums2) {
				return nums1[i1+k-1]
			}
			if k == 1 {
				return int(math.Min(float64(nums1[i1]), float64(nums2[i2])))
			}
			half := k >> 1
			newI1 := int(math.Min(float64(i1+half), float64(len(nums1)))) - 1
			newI2 := int(math.Min(float64(i2+half), float64(len(nums2)))) - 1
			if nums1[newI1] <= nums2[newI2] { // 不断舍弃小的那一半，永远不符合条件
				k -= newI1 - i1 + 1
				i1 = newI1 + 1
			} else {
				k -= newI2 - i2 + 1
				i2 = newI2 + 1
			}
		}
	}
	n := len(nums1) + len(nums2)
	if n%2 != 0 { // 奇
		return float64(getKthElement(nums1, nums2, n/2+1))
	} else { // 偶
		return float64(getKthElement(nums1, nums2, n/2)+getKthElement(nums1, nums2, n/2+1)) / 2
	}
}
func (m MyImplOne) generateParenthesis(n int) []string {
	var dfs func(ans []byte, l, r int)
	res := make([]string, 0)
	dfs = func(ans []byte, l, r int) {
		if l <= 0 && r <= 0 && len(ans) == 2*n {
			res = append(res, string(ans))
			return
		}
		if l > 0 {
			ans = append(ans, '(')
			dfs(ans, l-1, r)
			ans = ans[:len(ans)-1]
		}
		if r > 0 && r > l {
			ans = append(ans, ')')
			dfs(ans, l, r-1)
			ans = ans[:len(ans)-1]
		}
	}
	dfs(make([]byte, 0), n, n)
	return res
}

func (m MyImplOne) combinationSum(candidates []int, target int) [][]int {
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

func (m MyImplOne) sortList(head *ListNode) *ListNode {
	var dfs func(h *ListNode) *ListNode
	dfs = func(h *ListNode) *ListNode {
		if h == nil || h.Next == nil {
			return h
		}
		l, r := h, h.Next
		for r != nil && r.Next != nil {
			l = l.Next
			r = r.Next.Next
		}
		r = l.Next
		l.Next = nil
		h1 := dfs(h)
		h2 := dfs(r)
		a, b := h1, h2
		pivot := &ListNode{
			Val:  -1,
			Next: nil,
		}
		cur := pivot
		for a != nil && b != nil {
			if a.Val < b.Val {
				cur.Next = a
				a = a.Next
			} else {
				cur.Next = b
				b = b.Next
			}
			cur = cur.Next
		}
		if a != nil {
			cur.Next = a
		} else {
			cur.Next = b
		}
		return pivot.Next
	}
	return dfs(head)
}

func (m MyImplOne) coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i < len(dp); i++ {
		for _, c := range coins {
			if i >= c {
				dp[i] = int(math.Min(float64(dp[i]), float64(dp[i-c]+1)))
			}
		}
	}
	if dp[amount] <= amount {
		return dp[amount]
	} else {
		return -1
	}
}
func (m MyImplOne) widthOfBinaryTree(root *TreeNode) int {
	q := make([]*TreeNode, 0)
	root.Val = 0
	q = append(q, root)
	res := 0
	for len(q) != 0 {
		size := len(q)
		wide := q[len(q)-1].Val - q[0].Val + 1
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				node.Left.Val = 2*node.Val + 1
				q = append(q, node.Left)
			}
			if node.Right != nil {
				node.Right.Val = 2*node.Val + 2
				q = append(q, node.Right)
			}
		}
		res = int(math.Max(float64(res), float64(wide)))
	}
	return res
}
func (m MyImplOne) reverseBetween(head *ListNode, left int, right int) *ListNode {
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	pre, cur, tar := pivot, head, head
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

func (m MyImplOne) levelOrder(root *TreeNode) [][]int {

}
