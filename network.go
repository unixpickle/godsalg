package godsalg

import (
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weightnorm"
)

const (
	minScale        = 1
	maxScale        = 30
	sparseInitCount = 30
	moveCount       = 18
)

func CreateNetwork() neuralnet.Network {
	data, err := ioutil.ReadFile(os.Args[1])
	if err == nil {
		log.Println("Using existing network.")
		net, err := serializer.DeserializeWithType(data)
		if err != nil {
			panic(err)
		}
		return net.(neuralnet.Network)
	}

	log.Println("Creating new network.")

	return neuralnet.Network{
		weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(6*6*8, 1000)),
		&neuralnet.Sigmoid{},
		varyingFreqLayer(minScale, maxScale, 1000, 500),
		&neuralnet.Sin{},
		weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(500, 500)),
		&neuralnet.HyperbolicTangent{},
		weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(500, moveCount)),
		&neuralnet.LogSoftmaxLayer{},
	}
}

func varyingFreqLayer(minScale, maxScale float64, in, out int) neuralnet.Layer {
	res := weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(in, out))
	mags := res.Mags[0]
	for i := range mags.Vector {
		scale := minScale + (maxScale-minScale)*rand.Float64()
		mags.Vector[i] *= scale
	}

	// Sparse initializations will hopefully allow us to
	// utilize periodic values better.
	weights := res.Weights[0]
	for row := 0; row < out; row++ {
		rowVec := weights.Vector[row*in : (row+1)*in]
		for i := range rowVec {
			rowVec[i] = 0
		}
		for _, i := range rand.Perm(in)[:sparseInitCount] {
			rowVec[i] = rand.NormFloat64() / math.Sqrt(sparseInitCount)
		}
	}

	return res
}
