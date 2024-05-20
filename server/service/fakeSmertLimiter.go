package service

import (
	"bufio"
	"fmt"
	"os"
)

type sm struct {
	Val int
}

func (*sm) fake() {
	file, err := os.OpenFile("/Users/yehaitao/Desktop/workspace/fake_smartlimiter_config/sl.yml", os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		fmt.Printf("打开文件失败", err)
		return
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	instanceName := "ecs-o8egcxpu"
	for i := 0; i < 100; i++ {
		//instanceName := fmt.Sprintf("stargate.mytestmesh%d.xinye.com.g0.0", i)
		writer.WriteString(fmt.Sprintf(`---
apiVersion: microservice.slime.io/v1alpha2
kind: SmartLimiter
metadata:
  name: %s-circuitbreaker-18468-%d
  namespace: default-ops-mesh
spec:
  instance: true
  sets:
    _base:
      descriptor:
        - action:
            fill_interval:
              seconds: 65535
            fractionalPercent: 50
            quota: '1'
            resourceName: ptcheck2.ppdapi.com/api/mock/get
            strategy: single
          condition: 'true'
          target:
            direction: outbound
  workloadSelector:
    instance: %s
`, instanceName, i, instanceName))
		writer.WriteString("\n")
	}
	writer.Flush()
}
