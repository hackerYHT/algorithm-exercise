package system

import "github.com/hackerYHT/algorithm-exercise/server/service"

type ApiGroup struct {
	SolutionApi
}

var (
	solutionService = service.ServiceGroupApp.LcServiceGroup
)
