package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	_ "github.com/unixpickle/weightnorm"
)

const BatchSize = 30

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: solve <network>")
		os.Exit(1)
	}
	data, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read:", err)
		os.Exit(1)
	}
	obj, err := serializer.DeserializeWithType(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to deserialize:", err)
		os.Exit(1)
	}
	net := obj.(neuralnet.Network)
	cube, err := gocube.InputStickerCube()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Bad input:", err)
		os.Exit(1)
	}
	state, err := cube.CubieCube()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Bad state:", err)
		os.Exit(1)
	}
	for i := 0; true; i++ {
		solution := sampleSolution(*state, net)
		if solution != nil {
			fmt.Println("Solution:", solution)
			break
		} else {
			fmt.Println("Attempt", i, "failed")
		}
	}
}

func sampleSolution(start gocube.CubieCube, net neuralnet.Network) []gocube.Move {
	solutions := make([][]gocube.Move, BatchSize)
	states := make([]*gocube.CubieCube, BatchSize)
	for i := range states {
		c := start
		states[i] = &c
	}
	for i := 0; i < 21; i++ {
		var inVec linalg.Vector
		for j, x := range states {
			if x.Solved() {
				return solutions[j]
			}
			inVec = append(inVec, godsalg.CubeVector(x)...)
		}
		output := net.BatchLearner().Batch(&autofunc.Variable{Vector: inVec}, BatchSize)
		parts := autofunc.Split(BatchSize, output)
		for j, part := range parts {
			move := selectMoveVector(part.Output())
			solutions[j] = append(solutions[j], move)
			states[j].Move(move)
		}
	}
	return nil
}

func selectMoveVector(vec linalg.Vector) gocube.Move {
	p := rand.Float64()
	for i, x := range vec {
		p -= math.Exp(x)
		if p < 0 {
			return gocube.Move(i)
		}
	}
	return 0
}
