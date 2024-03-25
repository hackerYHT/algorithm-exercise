package study

import (
	"fmt"
	"math"
)

//多态

type Shape interface {
	Area() float64
}

type Rectangle struct {
	Width  float64
	Height float64
}

type Circle struct {
	Radius float64
}

func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

func (c Circle) Area() float64 {
	return math.Pi * math.Pow(c.Radius, 2)
}

func (c Circle) express() {
	shapes := []Shape{
		Rectangle{Width: 2, Height: 3},
		Circle{Radius: 1},
	}

	for _, shape := range shapes {
		fmt.Println(shape.Area())
	}
}
