package service

import (
	"github.com/hackerYHT/algorithm-exercise/server/service/lc"
)

type ServiceGroup struct {
	LcServiceGroup lc.ServiceGroup
}

var ServiceGroupApp = new(ServiceGroup)
