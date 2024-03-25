package main

import "github.com/hackerYHT/algorithm-exercise/server/core"

func main() {
	//r := gin.Default()
	//r.GET("/hs", func(c *gin.Context) { c.JSON(200, gin.H{"code": 200, "msg": "ok"}) })
	//r.Run(":8087")
	core.RunWindowsServer()
}
