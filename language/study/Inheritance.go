package study

import "fmt"

//继承

type Animal struct {
	Name string
}

func (a *Animal) Speak() {
	fmt.Println("Animal speaks")
}

type Dog struct {
	Animal // Animal结构体的嵌入
	Breed  string
}

func (dog Dog) express() {
	d := Dog{
		Animal: Animal{Name: "Dog"},
		Breed:  "Labrador",
	}

	fmt.Println(d.Name) // 可以直接访问Animal中的字段
	d.Speak()           // 可以调用Animal中的方法
}
