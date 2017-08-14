package main

import (
	"fmt"
	"os"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	_ "github.com/unixpickle/anyplugin"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/serializer"
)

const (
	MaxMoves = 16
)

func main() {
	if len(os.Args) != 2 {
		die("Usage: stats <network>")
	}
	var net anynet.Net
	if err := serializer.LoadAny(os.Args[1], &net); err != nil {
		die("Failed to load network:", err)
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

func roundOfSolves(net anynet.Net) []bool {
	cubes := make([]*gocube.CubieCube, MaxMoves)
	for i := range cubes {
		scramble, _ := godsalg.RandomScramble(i + 1)
		cubes[i] = scramble
	}
	res := make([]bool, MaxMoves)
	for i := 0; i < 21; i++ {
		var in []float64
		for j, c := range cubes {
			res[j] = res[j] || c.Solved()
			in = append(in, godsalg.CubeVector(c)...)
		}
		c := anyvec32.CurrentCreator()
		inRes := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(in)))
		out := net.Apply(inRes, MaxMoves)
		for j := 0; j < MaxMoves; j++ {
			subVec := out.Output().Slice(godsalg.NumMoves*j, godsalg.NumMoves*(j+1))
			max := anyvec.MaxIndex(subVec)
			cubes[j].Move(gocube.Move(max))
		}
	}
	return res
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
