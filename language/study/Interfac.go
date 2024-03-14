package study

import "fmt"

//接口
//只要一个类型实现了接口中定义的所有方法，就被视为实现了该接口

type Animal_1 interface {
	Speak()
}

type Dog_1 struct {
	Name string
}

func (d Dog_1) Speak() {
	fmt.Println("Dog barks")
}

func main() {
	var a Animal_1
	a = Dog_1{Name: "Max"}
	a.Speak() // 输出："Dog barks"
}
