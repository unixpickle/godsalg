package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weightnorm"
)

const (
	StepSize  = 1e-4
	BatchSize = 100

	OutputCount = 21
	MinMoves    = 9
	MaxMoves    = 16

	MinScale = 1
	MaxScale = 30

	SparseInitCount = 30
)

func init() {
	t := sinLayer{}.SerializerType()
	serializer.RegisterTypedDeserializer(t, deserializeSinLayer)
}

type DataPoint struct {
	Cube  *gocube.CubieCube
	Moves int
}

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	net := CreateNetwork()
	g := &sgd.Adam{Gradienter: &neuralnet.BatchRGradienter{
		Learner:      net.BatchLearner(),
		CostFunc:     neuralnet.DotCost{},
		MaxBatchSize: BatchSize,
	}}

	log.Println("Creating samples...")
	s := DataToVectors(GenerateData(1000000))
	log.Println("Training...")

	var iter int
	var last sgd.SampleSet
	sgd.SGDMini(g, s, StepSize, BatchSize, func(batch sgd.SampleSet) bool {
		if iter%4 == 0 {
			var lastCost float64
			if last != nil {
				lastCost = neuralnet.TotalCost(neuralnet.DotCost{}, net, last)
			}
			cost := neuralnet.TotalCost(neuralnet.DotCost{}, net, batch)
			lastCost /= BatchSize
			cost /= BatchSize
			log.Printf("iter %d: cost=%f last=%f", iter, cost, lastCost)
			last = batch.Copy()
		}
		iter++
		return true
	})

	file := os.Args[1]
	data, _ := serializer.SerializeWithType(net)
	if err := ioutil.WriteFile(file, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

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
		&neuralnet.HyperbolicTangent{},
		varyingFreqLayer(MinScale, MaxScale, 1000, 500),
		&sinLayer{},
		weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(500, 500)),
		&neuralnet.HyperbolicTangent{},
		weightnorm.NewDenseLayer(neuralnet.NewDenseLayer(500, OutputCount)),
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
		for _, i := range rand.Perm(in)[:SparseInitCount] {
			rowVec[i] = rand.NormFloat64() / math.Sqrt(SparseInitCount)
		}
	}

	return res
}

func GenerateData(count int) []DataPoint {
	var res []DataPoint
	for i := 0; i < count; i++ {
		moves := rand.Intn(MaxMoves-MinMoves+1) + MinMoves
		res = append(res, DataPoint{
			Cube:  RandomScramble(moves),
			Moves: moves,
		})
	}
	return res
}

func DataToVectors(d []DataPoint) sgd.SampleSet {
	inputs := make([]linalg.Vector, 0, len(d))
	outputs := make([]linalg.Vector, 0, len(d))
	for _, x := range d {
		inputs = append(inputs, CubeVector(x.Cube))
		vec := make(linalg.Vector, OutputCount)
		vec[x.Moves] = 1
		outputs = append(outputs, vec)
	}
	return neuralnet.VectorSampleSet(inputs, outputs)
}

func ClassifierScore(r neuralnet.Network, data []DataPoint) float64 {
	var numRight int
	var numTotal int
	for _, test := range data {
		guess := ClassifyCube(r, test.Cube)
		if guess == test.Moves {
			numRight++
		}
		numTotal++
	}
	return 100 * float64(numRight) / float64(numTotal)
}

func ClassifyCube(r neuralnet.Network, c *gocube.CubieCube) int {
	output := r.Apply(&autofunc.Variable{CubeVector(c)}).Output()
	var maxIdx int
	var maxVal float64
	for i, x := range output {
		if x > maxVal || i == 0 {
			maxIdx = i
			maxVal = x
		}
	}
	return maxIdx
}

type sinLayer struct {
	autofunc.Sin
}

func deserializeSinLayer(d []byte) (*sinLayer, error) {
	return &sinLayer{}, nil
}

func (_ sinLayer) SerializerType() string {
	return "github.com/unixpickle/lightsout.sinLayer"
}

func (_ sinLayer) Serialize() ([]byte, error) {
	return nil, nil
}
