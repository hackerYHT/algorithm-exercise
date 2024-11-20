package lru_2

import "container/list"

type LRUCache struct {
	capacity int
	lru_list *list.List
	hmap     map[int]*list.Element
}

type Entry struct {
	Key   int
	Value int
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		capacity: capacity,
		lru_list: list.New(),
		hmap:     make(map[int]*list.Element, 0),
	}
}

func (this *LRUCache) Get(key int) int {
	node, ok := this.hmap[key]
	if ok {
		this.lru_list.MoveToFront(node)
		return node.Value.(*Entry).Value
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	node, ok := this.hmap[key]
	if ok {
		node.Value.(*Entry).Value = value
		this.lru_list.MoveToFront(node)
	} else {
		this.hmap[key] = this.lru_list.PushFront(&Entry{
			Key:   key,
			Value: value,
		})
		if this.lru_list.Len() > this.capacity {
			delete(this.hmap, this.lru_list.Remove(this.lru_list.Back()).(*Entry).Key)
		}
	}

}
