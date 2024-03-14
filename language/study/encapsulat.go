package study

//封装
//name和age字段使用小写字母开头，表示私有字段。SetName()和GetName()方法是公共方法，可以通过它们来设置和获取私有字段

type Person struct {
	name string // 私有字段
	age  int    // 私有字段
}

func (p *Person) SetName(name string) {
	p.name = name
}

func (p *Person) GetName() string {
	return p.name
}
