package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/cuda"
	"github.com/unixpickle/godsalg"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	BatchSize = 10000
	StepSize  = 1e-5

	MinMoves  = 1
	MaxMoves  = 16
	MoveCount = 18
)

func main() {
	handle, err := cuda.NewHandle()
	if err != nil {
		panic(err)
	}
	anyvec32.Use(cuda.NewCreator32(handle))
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	c := anyvec32.CurrentCreator()
	net := godsalg.CreateNetwork(c, os.Args[1])

	log.Println("Training...")
	t := &anyff.Trainer{
		Net:     net,
		Cost:    anynet.DotCost{},
		Params:  net.Parameters(),
		Average: true,
	}

	var iterNum int
	s := &anysgd.SGD{
		Fetcher:     &Fetcher{Creator: c},
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     DummySampleList(BatchSize),
		Rater:       anysgd.ConstRater(StepSize),
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iterNum, t.LastCost)
			iterNum++
		},
		BatchSize: BatchSize,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	file := os.Args[1]
	if err := serializer.SaveAny(file, net); err != nil {
		fmt.Fprintln(os.Stderr, "Save error:", err)
		os.Exit(1)
	}
}

type DummySampleList int

func (d DummySampleList) Len() int {
	return int(d)
}

func (d DummySampleList) Swap(i, j int) {
}

func (d DummySampleList) Slice(i, j int) anysgd.SampleList {
	return DummySampleList(j - i)
}

type Fetcher struct {
	Creator anyvec.Creator
}

func (f *Fetcher) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	var inVec []float64
	var outVec []float64
	for i := 0; i < int(s.(DummySampleList)); i++ {
		moves := rand.Intn(MaxMoves-MinMoves+1) + MinMoves
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
		Num: int(s.(DummySampleList)),
	}, nil
}
