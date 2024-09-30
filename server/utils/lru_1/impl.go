package lru_1

import (
	"container/list"
)

type LRUCache struct {
	lru_list *list.List
	lru_map  map[int]*list.Element
	capacity int
}

type Entry struct {
	Key   int
	Value int
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		lru_list: list.New(),
		lru_map:  map[int]*list.Element{},
		capacity: capacity,
	}
}

func (this *LRUCache) Get(key int) int {
	v, ok := this.lru_map[key]
	if ok {
		this.lru_list.MoveToFront(v)
		return v.Value.(*Entry).Value
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	node, ok := this.lru_map[key]
	new_entry := &Entry{
		Key:   key,
		Value: value,
	}
	if ok {
		node.Value = new_entry
		this.lru_list.MoveToFront(node)
	} else {
		this.lru_map[key] = this.lru_list.PushFront(new_entry)
		if len(this.lru_map) > this.capacity {
			delete(this.lru_map, this.lru_list.Remove(this.lru_list.Back()).(*Entry).Key)
		}
	}
}
