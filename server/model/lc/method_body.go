package lc

import (
	"github.com/flipped-aurora/gin-vue-admin/server/global"
)

type MethodBody struct {
	global.GVA_MODEL
	Method string `json:"method" gorm:"方法名称"` // a方法名称
}
