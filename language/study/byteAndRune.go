package study

import "fmt"

//byte和rune有什么区别

func Express() {

	var b byte = 'a'
	fmt.Printf("%c\n", b) // 输出: a
	var r rune = '世'
	fmt.Printf("%c\n", r) // 输出: 世

}
