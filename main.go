package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"sync/atomic"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	StepSize     = 1e-6
	BigBatchSize = 4000
	BatchSize    = 120

	ValidationCount = 200

	OutputCount = 21
	MinMoves    = 9
	MaxMoves    = 16
)

var HiddenSizes = []int{3000, 2000}

type DataPoint struct {
	Cube  *gocube.CubieCube
	Moves int
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	net := CreateNetwork()
	batcher := &neuralnet.BatchRGradienter{
		Learner:      net.BatchLearner(),
		CostFunc:     neuralnet.MeanSquaredCost{},
		MaxBatchSize: 15,
	}
	rms := &neuralnet.RMSProp{Gradienter: batcher}

	signal := KillSignal()

	trainingData := GenerateData(BigBatchSize)
	for atomic.LoadUint32(signal) == 0 {
		samples := DataToVectors(trainingData)
		neuralnet.SGD(rms, samples, StepSize, 1, BigBatchSize)
		trainingData = GenerateData(BigBatchSize)
		t := trainingData[:ValidationCount]
		log.Printf("Training success: %.02f%%", ClassifierScore(net, t))
	}

	file := os.Args[1]
	data, _ := net.Serialize()
	if err := ioutil.WriteFile(file, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func CreateNetwork() neuralnet.Network {
	net := neuralnet.Network{}
	for i, x := range HiddenSizes {
		inputSize := 6 * 6 * 8
		if i > 0 {
			inputSize = HiddenSizes[i-1]
		}
		layer := &neuralnet.DenseLayer{
			InputCount:  inputSize,
			OutputCount: x,
		}
		net = append(net, layer, neuralnet.Sigmoid{})
	}
	layer := &neuralnet.DenseLayer{
		InputCount:  HiddenSizes[len(HiddenSizes)-1],
		OutputCount: OutputCount,
	}
	net = append(net, layer, &neuralnet.SoftmaxLayer{})
	net.Randomize()
	return net
}

func KillSignal() *uint32 {
	var killFlag uint32
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
		atomic.StoreUint32(&killFlag, 1)
	}()
	return &killFlag
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

func DataToVectors(d []DataPoint) *neuralnet.SampleSet {
	samples := &neuralnet.SampleSet{
		Inputs:  make([]linalg.Vector, 0, len(d)),
		Outputs: make([]linalg.Vector, 0, len(d)),
	}
	for _, x := range d {
		samples.Inputs = append(samples.Inputs, CubeVector(x.Cube))
		vec := make(linalg.Vector, OutputCount)
		vec[x.Moves] = 1
		samples.Outputs = append(samples.Outputs, vec)
	}
	return samples
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
		if x > maxVal {
			maxIdx = i
			maxVal = x
		}
	}
	return maxIdx
}
