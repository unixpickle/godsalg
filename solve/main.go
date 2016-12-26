package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	_ "github.com/unixpickle/weightnorm"
)

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
	for i := 0; i < 1000; i++ {
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
	var solution []gocube.Move
	for i := 0; i < 21; i++ {
		if start.Solved() {
			return solution
		}
		vec := CubeVector(&start)
		output := net.Apply(&autofunc.Variable{Vector: vec}).Output()
		move := selectMoveVector(output)
		solution = append(solution, move)
		start.Move(move)
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

// CubeVector returns a vectorized representation of
// the stickers of a cube.
func CubeVector(c *gocube.CubieCube) linalg.Vector {
	stickerCube := c.StickerCube()
	res := make(linalg.Vector, 8*6*6)

	var stickerIdx int
	for i, sticker := range stickerCube[:] {
		if i%9 == 4 {
			continue
		}
		for j := 0; j < 6; j++ {
			if j == sticker-1 {
				res[j+stickerIdx] = 1.0
			} else {
				res[j+stickerIdx] = -0.2
			}
		}
		stickerIdx += 6
	}

	return res
}
