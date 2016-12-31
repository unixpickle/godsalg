package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	MaxMoves = 16
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
		solves := roundOfSolves(net)
		for i, x := range solves {
			if x {
				histogram[i]++
			}
			total[i]++
		}
		for i := 1; i <= MaxMoves; i++ {
			pct := histogram[i-1] / total[i-1]
			fmt.Println(i, "moves:", pct*100, "%")
		}
	}
}

func roundOfSolves(net neuralnet.Network) []bool {
	cubes := make([]*gocube.CubieCube, MaxMoves)
	for i := range cubes {
		scramble, _ := godsalg.RandomScramble(i + 1)
		cubes[i] = scramble
	}
	res := make([]bool, MaxMoves)
	for i := 0; i < 21; i++ {
		var in linalg.Vector
		for j, c := range cubes {
			res[j] = res[j] || c.Solved()
			in = append(in, godsalg.CubeVector(c)...)
		}
		inRes := &autofunc.Variable{Vector: in}
		out := net.BatchLearner().Batch(inRes, MaxMoves)
		for j, moveVec := range autofunc.Split(MaxMoves, out) {
			_, move := moveVec.Output().Max()
			cubes[j].Move(gocube.Move(move))
		}
	}
	return res
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
