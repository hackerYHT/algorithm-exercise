package v1

import lc "github.com/hackerYHT/algorithm-exercise/server/api/v1/lc"

type ApiGroup struct {
	LcApiGroup lc.ApiGroup
}

var ApiGroupApp = new(ApiGroup)
