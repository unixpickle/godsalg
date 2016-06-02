package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/signal"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/lstm"
	"github.com/unixpickle/weakai/rnn/softmax"
)

const (
	HiddenSize1 = 500
	HiddenSize2 = 500
	StepSize    = 0.00001
	BatchSize   = 100

	MinMoves = 9
	MaxMoves = 16

	TrainingCount   = 40000
	ValidationCount = 500
)

type DataPoint struct {
	Cube  *gocube.CubieCube
	Moves int
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	net := rnn.DeepRNN{
		lstm.NewNet(rnn.ReLU{}, 6*6*8, 0, HiddenSize1),
		lstm.NewNet(rnn.ReLU{}, HiddenSize1, 0, HiddenSize2),
		lstm.NewNet(rnn.ReLU{}, HiddenSize2, 0, 21),
		softmax.NewSoftmax(21),
	}
	net.Randomize()

	trainingData := GenerateData(TrainingCount)

	inVecs, outVecs := DataToVectors(trainingData)
	trainer := rnn.RMSProp{
		SGD: rnn.SGD{
			CostFunc:  rnn.MeanSquaredCost{},
			InSeqs:    inVecs,
			OutSeqs:   outVecs,
			StepSize:  StepSize,
			BatchSize: BatchSize,
			Epochs:    1,
		},
	}

	killChan := make(chan struct{})

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
		close(killChan)
	}()

	testingData := GenerateData(ValidationCount)
TrainLoop:
	for {
		select {
		case <-killChan:
			break TrainLoop
		default:
		}
		log.Printf("Score: training=%.02f%%, cross=%.02f%%", ClassifierScore(net, trainingData),
			ClassifierScore(net, testingData))
		select {
		case <-killChan:
			break TrainLoop
		default:
		}
		trainer.Train(net)
	}

	file := os.Args[1]
	data, _ := net.Serialize()
	if err := ioutil.WriteFile(file, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
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

func DataToVectors(d []DataPoint) (v1, v2 [][]linalg.Vector) {
	for _, x := range d {
		v1 = append(v1, []linalg.Vector{CubeVector(x.Cube)})
		vec := make(linalg.Vector, 21)
		vec[x.Moves] = 1
		v2 = append(v2, []linalg.Vector{vec})
	}
	return
}

func ClassifierScore(r rnn.RNN, data []DataPoint) float64 {
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

func ClassifyCube(r rnn.RNN, c *gocube.CubieCube) int {
	output := r.StepTime(CubeVector(c))
	r.Reset()
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
