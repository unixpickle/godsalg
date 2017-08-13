package godsalg

import (
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anymisc"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

const (
	moveCount = 18
)

func CreateNetwork(c anyvec.Creator, path string) anynet.Net {
	var net anynet.Net
	if err := serializer.LoadAny(path, &net); err == nil {
		log.Println("Using existing network.")
		return net
	}

	log.Println("Creating new network...")
	res := anynet.Net{
		anynet.NewFC(c, 6*6*8, 1024),
		&anymisc.SELU{},
	}
	for i := 0; i < 30; i++ {
		res = append(res, anynet.NewFC(c, 1024, 1024), &anymisc.SELU{})
	}
	res = append(res, anynet.NewFC(c, 1024, moveCount), anynet.LogSoftmax)
	return res
}
