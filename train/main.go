package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	StepSize    = 1e-4
	BatchSize   = 100
	SampleCount = 1000000
	LogInterval = 64

	MinMoves  = 1
	MaxMoves  = 16
	MoveCount = 18
)

type DataPoint struct {
	Cube  *gocube.CubieCube
	First gocube.Move
}

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	net := godsalg.CreateNetwork()
	g := &sgd.Adam{Gradienter: &neuralnet.BatchRGradienter{
		Learner:      net.BatchLearner(),
		CostFunc:     neuralnet.DotCost{},
		MaxBatchSize: BatchSize,
	}}

	log.Println("Creating samples...")
	s := DataToVectors(GenerateData(SampleCount))
	log.Println("Training...")

	var iter int
	var last sgd.SampleSet
	sgd.SGDMini(g, s, StepSize, BatchSize, func(batch sgd.SampleSet) bool {
		if iter%LogInterval == 0 {
			var lastCost float64
			bl := net.BatchLearner()
			if last != nil {
				lastCost = neuralnet.TotalCostBatcher(neuralnet.DotCost{}, bl, last, 0)
			}
			cost := neuralnet.TotalCostBatcher(neuralnet.DotCost{}, bl, batch, 0)
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

func GenerateData(count int) []DataPoint {
	var res []DataPoint
	for i := 0; i < count; i++ {
		moves := rand.Intn(MaxMoves-MinMoves+1) + MinMoves
		cube, first := godsalg.RandomScramble(moves)
		res = append(res, DataPoint{
			Cube:  cube,
			First: first,
		})
	}
	return res
}

func DataToVectors(d []DataPoint) sgd.SampleSet {
	inputs := make([]linalg.Vector, 0, len(d))
	outputs := make([]linalg.Vector, 0, len(d))
	for _, x := range d {
		inputs = append(inputs, godsalg.CubeVector(x.Cube))
		vec := make(linalg.Vector, MoveCount)
		vec[x.First] = 1
		outputs = append(outputs, vec)
	}
	return neuralnet.VectorSampleSet(inputs, outputs)
}
