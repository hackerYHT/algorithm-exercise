package lc

import (
	"github.com/gin-gonic/gin"
	v1 "github.com/hackerYHT/algorithm-exercise/server/api/v1"
)

type LcGroup struct {
	SolutionRouter
}

func (g LcGroup) InitBaseRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	baseRouter := Router.Group("lc")
	solutionApi := v1.ApiGroupApp.LcApiGroup.SolutionApi
	{
		baseRouter.POST("algorithm", solutionApi.Resolve)
		baseRouter.POST("convert", solutionApi.Convert)
	}
	return baseRouter

}
