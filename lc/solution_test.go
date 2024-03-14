package lc

import (
	"reflect"
	"testing"
)

func TestMyImpl_groupAnagrams(t *testing.T) {
	type fields struct {
		Algorithm Algorithm
		Name      string
	}
	type args struct {
		strs []string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   [][]string
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := MyImpl{
				Algorithm: tt.fields.Algorithm,
				Name:      tt.fields.Name,
			}
			if got := m.groupAnagrams(tt.args.strs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("groupAnagrams() = %v, want %v", got, tt.want)
			}
		})
	}
}
