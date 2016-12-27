package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	MaxMoves    = 16
	LogInterval = 10
)

func main() {
	if len(os.Args) != 2 {
		die("Usage: stats <network>")
	}
	netData, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		die("Failed to read network:", err)
	}
	obj, err := serializer.DeserializeWithType(netData)
	if err != nil {
		die("Failed to deserialize network:", err)
	}
	net, ok := obj.(neuralnet.Network)
	if !ok {
		die("Not a neuralnet.Network")
	}

	histogram := make([]float64, MaxMoves)
	total := make([]float64, MaxMoves)
	for i := 0; true; i++ {
		moves := (i % MaxMoves) + 1
		scramble, _ := godsalg.RandomScramble(moves)
		total[moves-1]++
		if solves(net, *scramble) {
			histogram[moves-1]++
		}
		if i%LogInterval == 0 {
			for i := 1; i <= MaxMoves; i++ {
				pct := histogram[i-1] / total[i-1]
				fmt.Println(i, "moves:", pct*100, "%")
			}
		}
	}
}

func solves(net neuralnet.Network, cube gocube.CubieCube) bool {
	for i := 0; i < 21; i++ {
		if cube.Solved() {
			return true
		}
		vec := godsalg.CubeVector(&cube)
		output := net.Apply(&autofunc.Variable{Vector: vec}).Output()
		_, move := output.Max()
		cube.Move(gocube.Move(move))
	}
	return false
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
