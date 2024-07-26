package utils

import "container/list"

type LRUCatch struct {
	cap     int
	linkLst *list.List
	lruMap  map[int]*list.Element
}

type entry struct {
	key   int
	value int
}

func Constructer(cap int) LRUCatch {
	return LRUCatch{
		cap:     cap,
		linkLst: list.New(),
		lruMap:  map[int]*list.Element{},
	}
}
func (c *LRUCatch) Get(key int) int {
	node, ok := c.lruMap[key]
	if ok {
		c.linkLst.MoveToFront(node)
		return node.Value.(entry).value
	} else {
		return -1
	}
}

func (c *LRUCatch) Put(key, value int) {
	node, ok := c.lruMap[key]
	node.Value = entry{key, value}
	if ok {
		c.linkLst.MoveToFront(node)
	} else {
		c.linkLst.PushFront(node)
		if len(c.lruMap) > c.cap {
			delete(c.lruMap, c.linkLst.Remove(c.linkLst.Back()).(entry).key)
		}
	}
}
