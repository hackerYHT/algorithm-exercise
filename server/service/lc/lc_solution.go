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
	return impl.SolveNQueens(4)
}

func (s *ServiceGroup) Convert(request lc.ConvertRequest) any {
	return request.Result[0].Content
}
