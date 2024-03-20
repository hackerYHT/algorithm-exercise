package lc

import (
	"github.com/hackerYHT/algorithm-exercise/server/model/lc"
	"github.com/hackerYHT/algorithm-exercise/server/utils"
)

type SolutionService struct{}

type AlgorithmImpl struct {
	utils.MyImpl
}

func (s *ServiceGroup) Resolve(mb lc.MethodBody) any {
	var impl AlgorithmImpl
	return impl.SubarraySum([]int{1, 2, 3}, 3)
}
