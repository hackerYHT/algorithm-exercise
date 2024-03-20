package config

type Server struct {
	System System `mapstructure:"system" json:"system" yaml:"system"`
	Local  Local  `mapstructure:"local" json:"local" yaml:"local"`
}
