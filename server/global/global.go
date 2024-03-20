package global

import (
	"github.com/hackerYHT/algorithm-exercise/server/config"
	"go.uber.org/zap"
	"sync"
)

var (
	GVA_CONFIG config.Server
	GVA_LOG    *zap.Logger
	lock       sync.RWMutex
)
