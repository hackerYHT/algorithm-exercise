package utils

import (
	"math"
	"strings"
)

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

func (m MyImplThree) lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		if node.Val == p.Val || node.Val == q.Val {
			return node
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left != nil && right != nil {
			return node
		}
		if left != nil {
			return left
		}
		if right != nil {
			return right
		}
		return nil
	}
	return dfs(root)
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
		if x >= len {
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
			heapify(arr, i, len(arr))
		}
	}
	buildHeap(nums)
	for i := 0; i < k; i++ {
		heapify(nums, 0, len(nums)-i)
		nums[0], nums[len(nums)-i-1] = nums[len(nums)-i-1], nums[0]
	}
	return nums[len(nums)-k]
}
func (m MyImplThree) deleteNode(root *TreeNode, key int) *TreeNode {
	updateNode := func(node *TreeNode) *TreeNode {
		if node.Left == nil && node.Right == nil {
			return nil
		}
		if node.Left != nil && node.Right != nil {
			cur := node.Right
			for cur.Left != nil {
				cur = cur.Left
			}
			cur.Left = node.Left
			return node.Right
		}
		if node.Left != nil {
			return node.Left
		}
		if node.Right != nil {
			return node.Right
		}
		return nil
	}
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		if node.Val == key {
			return updateNode(node)
		}
		node.Left = dfs(node.Left)
		node.Right = dfs(node.Right)
		return node
	}
	return dfs(root)
}

func (m MyImplThree) maxPathSum(root *TreeNode) int {
	res := math.MinInt
	//返回经过root的单边分支最大和， 即Math.max(root, root+left, root+right)
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftMax := int(math.Max(float64(dfs(node.Left)), 0))
		rightMax := int(math.Max(float64(dfs(node.Right)), 0))
		if res < node.Val+leftMax+rightMax {
			res = node.Val + leftMax + rightMax
		}
		return node.Val + int(math.Max(float64(leftMax), float64(rightMax)))
	}
	dfs(root)
	return res
}
func (m MyImplThree) maxSubArray(nums []int) int {
	dp := make([]int, len(nums))
	res := nums[0]
	dp[0] = nums[0]
	for i := 1; i < len(dp); i++ {
		dp[i] = int(math.Max(float64(dp[i-1]+nums[i]), float64(nums[i])))
		if dp[i] > res {
			res = dp[i]
		}
	}
	return res
}

func (m MyImplThree) canFinish(numCourses int, prerequisites [][]int) bool {
	//统计图节点的入度
	degreeArr := make([]int, numCourses)
	//采用map表示领结矩阵（无向图）
	graphMap := make(map[int][]int)
	//初始化数据结构
	for i := 0; i < len(prerequisites); i++ {
		nextCourseArr, ok := graphMap[prerequisites[i][1]]
		if ok {
			//切片扩容触发深拷贝，不会同时修改map的值
			graphMap[prerequisites[i][1]] = append(nextCourseArr, prerequisites[i][0])
		} else {
			graphMap[prerequisites[i][1]] = []int{prerequisites[i][0]}
		}
		degreeArr[prerequisites[i][0]]++
	}
	//遍历数据结构，处理入度，从入度==0的图节点开始
	q := make([]int, 0)
	for i := 0; i < len(degreeArr); i++ {
		if degreeArr[i] == 0 {
			q = append(q, i)
		}
	}
	count := 0
	for len(q) != 0 {
		size := len(q)
		count += size
		for i := 0; i < size; i++ {
			nextCourseArr, ok := graphMap[q[0]]
			q = q[1:]
			if ok {
				for j := 0; j < len(nextCourseArr); j++ {
					degreeArr[nextCourseArr[j]]--
					if degreeArr[nextCourseArr[j]] == 0 {
						q = append(q, nextCourseArr[j])
					}
				}
			}
		}
	}
	if count != numCourses {
		return false
	}
	return true
}
func (m MyImplThree) reverseWords(s string) string {
	strArr := strings.Fields(s)
	sb := strings.Builder{}
	for i := len(strArr) - 1; i >= 0; i-- {
		sb.WriteString(strArr[i])
		if i != 0 {
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

func (m MyImplThree) subarraySum(nums []int, k int) int {
	dp := make([]int, len(nums))
	res := 0
	if nums[0] == k {
		dp[0]++
		res++
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
func (m MyImplThree) generateParenthesis(n int) []string {
	var dfs func(ans []byte, l, r int)
	res := make([]string, 0)
	dfs = func(ans []byte, l, r int) {
		if l == 0 && r == 0 {
			tmp := make([]byte, len(ans))
			copy(tmp, ans)
			res = append(res, string(tmp))
		}
		if l > 0 {
			ans = append(ans, '(')
			dfs(ans, l-1, r)
			ans = ans[:len(ans)-1]
		}
		if r > l && r > 0 {
			ans = append(ans, ')')
			dfs(ans, l, r-1)
			ans = ans[:len(ans)-1]
		}
	}
	dfs(make([]byte, 0), n, n)
	return res
}
func (m MyImplThree) findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	getTheKthNum := func(k int) float64 {
		i1, i2 := 0, 0
		for {
			if i1 == len(nums1) {
				return float64(nums2[i2+k-1])
			}
			if i2 == len(nums2) {
				return float64(nums1[i1+k-1])
			}
			if k == 1 {
				return math.Min(float64(nums1[i1]), float64(nums2[i2]))
			}
			half := k / 2
			newi1 := int(math.Min(float64(i1+half), float64(len(nums1)))) - 1
			newi2 := int(math.Min(float64(i2+half), float64(len(nums2)))) - 1
			if nums1[newi1] <= nums2[newi2] {
				k = k - (newi1 - i1 + 1)
				i1 = newi1 + 1
			} else {
				k = k - (newi2 - i2 + 1)
				i2 = newi2 + 1
			}
		}
	}
	len := len(nums1) + len(nums2)
	if len%2 == 0 {
		return (getTheKthNum(len/2+1) + getTheKthNum(len/2)) / 2
	} else {
		return getTheKthNum(len/2 + 1)
	}
}

func (m MyImplThree) rotate(matrix [][]int) {
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

//func (m MyImplThree) longestValidParentheses(s string) int {
//}

func (m MyImplThree) reverseKGroup(head *ListNode, k int) *ListNode {
	var reverseList func(pre, cur, tar *ListNode) *ListNode
	reverseList = func(pre, cur, tar *ListNode) *ListNode {
		if cur == nil || cur == tar {
			return cur
		}
		tmp := reverseList(cur, cur.Next, tar)
		cur.Next = pre
		return tmp
	}
	pivot := &ListNode{
		Val:  -1,
		Next: head,
	}
	start, end := head, head
	for i := 0; i < k; i++ {
		end = end.Next
	}
	for {
		reverseList(start, end, end.Next)
		for i := 0; i < k; i++ {
			end = end.Next
		}
	}
	return pivot.Next
}

func (m MyImplThree) reverseBetween(head *ListNode, left int, right int) *ListNode {
	//新方法，递归，翻转链表2
	var dfs func(cur, pre, tar *ListNode) *ListNode
	dfs = func(cur, pre, tar *ListNode) *ListNode {
		if cur == nil || cur == tar {
			return pre
		}
		tmp := dfs(cur.Next, cur, tar)
		cur.Next = pre
		return tmp
	}
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
	v := dfs(cur, tar, tar)
	pre.Next = v
	return pivot.Next
}
