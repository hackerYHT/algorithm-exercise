package utils

import "math"

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
