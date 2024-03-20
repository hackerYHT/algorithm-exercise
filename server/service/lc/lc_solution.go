package lc

import (
	"github.com/hackerYHT/algorithm-exercise/server/model/lc"
	"github.com/hackerYHT/algorithm-exercise/server/utils"
	"reflect"
)

type SolutionService struct{}

type AlgorithmImpl struct {
	utils.MyImpl
}

func (s *ServiceGroup) Resolve(mb lc.MethodBody) any {
	method := mb.Method
	var impl utils.MyImpl
	// 使用反射遍历方法
	methodValue := reflect.ValueOf(&impl).Elem()
	for i := 0; i < methodValue.NumMethod(); i++ {
		methodName := methodValue.Type().Method(i).Name
		if methodName == method {
			// 找到匹配的方法，执行并返回结果
			result := methodValue.Method(i).Call(nil)
			return result[0].Interface()
		}
	}
	return nil
}
