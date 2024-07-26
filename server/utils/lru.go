package utils

import "container/list"

type LRUCache struct {
	cap     int
	linkLst *list.List
	lruMap  map[int]*list.Element
}

type entry struct {
	key   int
	value int
}

func Constructor(cap int) LRUCache {
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
