package godsalg

import (
	"log"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

const (
	maxScale  = 30
	moveCount = 18
)

func init() {
	s := SinLayer{}
	t := s.SerializerType()
	serializer.RegisterTypedDeserializer(t, func(d []byte) (SinLayer, error) {
		return s, nil
	})
}

func CreateNetwork(c anyvec.Creator, path string) anynet.Net {
	var net anynet.Net
	if err := serializer.LoadAny(path, &net); err == nil {
		log.Println("Using existing network.")
		return net
	}

	log.Println("Creating new network...")
	return anynet.Net{
		anynet.NewFC(c, 6*6*8, 1024),
		periodScaler(c, 1024),
		SinLayer{},
		anynet.NewFC(c, 1024, 1024),
		periodScaler(c, 1024),
		SinLayer{},
		anynet.NewFC(c, 1024, 1024),
		periodScaler(c, 1024),
		SinLayer{},
		anynet.NewFC(c, 1024, 512),
		anynet.Tanh,
		anynet.NewFC(c, 512, moveCount),
		anynet.LogSoftmax,
	}
}

func periodScaler(c anyvec.Creator, inSize int) *anynet.Affine {
	layer := &anynet.Affine{
		Scalers: anydiff.NewVar(c.MakeVector(inSize)),
		Biases:  anydiff.NewVar(c.MakeVector(inSize)),
	}
	anyvec.Rand(layer.Scalers.Vector, anyvec.Uniform, nil)
	layer.Scalers.Vector.Scale(c.MakeNumeric(maxScale))
	return layer
}

type SinLayer struct{}

func (s SinLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	return anydiff.Sin(in)
}

func (s SinLayer) SerializerType() string {
	return "github.com/unixpickle/godsalg.SinLayer"
}

func (s SinLayer) Serialize() ([]byte, error) {
	return []byte{}, nil
}
