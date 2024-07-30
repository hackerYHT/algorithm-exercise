package utils

import (
	"container/list"
	"math"
	"sort"
)

type LRUCache struct {
	cap     int
	linkLst *list.List
	lruMap  map[int]*list.Element
}

type entry struct {
	key   int
	value int
}

func Construct(cap int) LRUCache {
	return LRUCache{
		cap:     cap,
		linkLst: list.New(),
		lruMap:  map[int]*list.Element{},
	}
}
func (c *LRUCache) Get(key int) int {
	node, ok := c.lruMap[key]
	if ok {
		c.linkLst.MoveToFront(node)
		return node.Value.(entry).value
	} else {
		return -1
	}
}

func (c *LRUCache) Put(key, value int) {
	node, ok := c.lruMap[key]
	if ok {
		node.Value = entry{key, value}
		c.linkLst.MoveToFront(node)
	} else {
		c.lruMap[key] = c.linkLst.PushFront(entry{key, value})
		if len(c.lruMap) > c.cap {
			delete(c.lruMap, c.linkLst.Remove(c.linkLst.Back()).(entry).key)
		}
	}
}
func (m MyImpl) merge(intervals [][]int) [][]int {
	res := make([][]int, 0)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	max := intervals[0][1]
	for x, y := 0, 0; y < len(intervals); {
		max = int(math.Max(float64(max), float64(intervals[y][1])))
		if y == len(intervals)-1 || max < intervals[y+1][0] {
			res = append(res, []int{intervals[x][0], max})
			x, y = y+1, y+1
		} else {
			y++
		}
	}
	return res
}
