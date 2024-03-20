package core

import (
	"fmt"
	"github.com/hackerYHT/algorithm-exercise/server/global"
	"github.com/hackerYHT/algorithm-exercise/server/initialize"
	"go.uber.org/zap"
)

type server interface {
	ListenAndServe() error
}

func RunWindowsServer() {

	Router := initialize.Routers()
	Router.Static("/form-generator", "./resource/page")

	address := fmt.Sprintf(":%d", global.GVA_CONFIG.System.Addr)
	s := initServer(address, Router)

	global.GVA_LOG.Info("server run success on ", zap.String("address", address))

	fmt.Printf(`
	当前版本:v1.0.0
	默认自动化文档地址:http://127.0.0.1%s/swagger/index.html
`, address)
	global.GVA_LOG.Error(s.ListenAndServe().Error())
}
