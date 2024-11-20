package lru_2

import "container/list"

type LRUCache struct {
	capacity int
	list     *list.List
	hmap     map[int]*list.Element
}

type Entry struct {
	key   int
	value int
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		capacity: capacity,
		list:     &list.List{},
		hmap:     make(map[int]*list.Element, 0),
	}
}

func (this *LRUCache) Get(key int) int {
	node, ok := this.hmap[key]
	if ok {
		this.list.MoveToFront(node)
		return node.Value.(*Entry).value
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	node, ok := this.hmap[key]
	if ok {
		node.Value.(*Entry).value = value
		this.list.PushFront(node)
	} else {
		if this.capacity >= this.list.Len() {
			this.list.Remove(this.list.Back())
			delete(this.hmap, key)
			this.capacity--
		}
		this.hmap[key] = this.list.PushFront(&Entry{
			key:   key,
			value: value,
		})
		this.capacity++
	}

}
