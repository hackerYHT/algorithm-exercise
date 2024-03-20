package core

import (
	"fmt"
	"github.com/hackerYHT/algorithm-exercise/server/initialize"
)

type server interface {
	ListenAndServe() error
}

func RunWindowsServer() {

	Router := initialize.Routers()
	Router.Static("/form-generator", "./resource/page")

	address := fmt.Sprintf(":%d", 8888)
	s := initServer(address, Router)
	fmt.Println(address)

	fmt.Printf(`
	当前版本:v1.0.0
	默认自动化文档地址:http://127.0.0.1%s/swagger/index.html
`, address)
	err := s.ListenAndServe()
	if err != nil {
		return
	}
}
