package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	_ "github.com/unixpickle/anyplugin"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	MoveCount = 18
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var outFile string
	var batchSize int
	var stepSize float64
	var minMoves int
	var maxMoves int

	flag.StringVar(&outFile, "out", "out_net", "output network file")
	flag.IntVar(&batchSize, "batch", 1000, "SGD batch size")
	flag.Float64Var(&stepSize, "step", 1e-4, "SGD step size")
	flag.IntVar(&minMoves, "minmoves", 1, "minimum scramble moves")
	flag.IntVar(&maxMoves, "maxmoves", 16, "maximum scramble moves")
	flag.Parse()

	c := anyvec32.CurrentCreator()
	net := godsalg.CreateNetwork(c, outFile)

	log.Println("Training...")
	t := &anyff.Trainer{
		Net:     net,
		Cost:    anynet.DotCost{},
		Params:  net.Parameters(),
		Average: true,
	}

	var iterNum int
	s := &anysgd.SGD{
		Fetcher: &Fetcher{
			Creator:  c,
			MinMoves: minMoves,
			MaxMoves: maxMoves,
		},
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     anysgd.LengthSampleList(batchSize),
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iterNum, t.LastCost)
			iterNum++
		},
		BatchSize: batchSize,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	if err := serializer.SaveAny(outFile, net); err != nil {
		fmt.Fprintln(os.Stderr, "Save error:", err)
		os.Exit(1)
	}
}

type Fetcher struct {
	Creator  anyvec.Creator
	MinMoves int
	MaxMoves int
}

func (f *Fetcher) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	var inVec []float64
	var outVec []float64
	for i := 0; i < s.Len(); i++ {
		moves := rand.Intn(f.MaxMoves-f.MinMoves+1) + f.MinMoves
		cube, first := godsalg.RandomScramble(moves)
		inVec = append(inVec, godsalg.CubeVector(cube)...)

		oneHot := make([]float64, MoveCount)
		oneHot[first] = 1
		outVec = append(outVec, oneHot...)
	}

	return &anyff.Batch{
		Inputs: anydiff.NewConst(
			f.Creator.MakeVectorData(f.Creator.MakeNumericList(inVec)),
		),
		Outputs: anydiff.NewConst(
			f.Creator.MakeVectorData(f.Creator.MakeNumericList(outVec)),
		),
		Num: s.Len(),
	}, nil
}
