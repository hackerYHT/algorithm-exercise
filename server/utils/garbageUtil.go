package utils

import (
	"fmt"
	"runtime"
)

func Test() {
	fmt.Println("程序开始时的内存占用：", getMemUsage())

	for i := 0; i < 10; i++ {
		createGarbage()
	}

	fmt.Println("初次创建垃圾后的内存占用：", getMemUsage())

	// 手动触发垃圾回收
	runtime.GC()

	fmt.Println("手动触发垃圾回收后的内存占用：", getMemUsage())
}

func createGarbage() {
	for i := 0; i < 10000; i++ {
		_ = make([]byte, 1024)
	}
}

func getMemUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}
