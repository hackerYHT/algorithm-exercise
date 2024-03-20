package router

import (
	"github.com/hackerYHT/algorithm-exercise/server/router/lc"
)

type RouterGroup struct {
	Lc lc.LcGroup
}

var RouterGroupApp = new(RouterGroup)
