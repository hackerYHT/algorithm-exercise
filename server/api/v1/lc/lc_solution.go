package system

import (
	"github.com/flipped-aurora/gin-vue-admin/server/model/common/response"
	"github.com/gin-gonic/gin"
	"github.com/hackerYHT/algorithm-exercise/server/model/lc"
)

type SolutionApi struct{}

func (s *SolutionApi) Resolve(c *gin.Context) {
	var mb lc.MethodBody
	err := c.ShouldBindJSON(&mb)
	if err != nil {
		response.FailWithMessage(err.Error(), c)
		return
	}
	if err != nil {
		response.FailWithMessage(err.Error(), c)
		return
	}
	res := solutionService.Resolve(mb)
	if res != nil {
		response.OkWithMessage("执行失败！", c)
	} else {
		response.OkWithDetailed(res, "执行成功！", c)
	}
}
