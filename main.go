package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/weakai/idtrees"
)

const (
	NumSamples = 300000
	NumTrees   = 1000
	SubSamples = 4000
	SubAttrs   = 30

	MinMoves = 5
	MaxMoves = 18
)

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	var features []idtrees.Attr
	for i := 0; i < DataFeatureCount; i++ {
		features = append(features, i)
	}
	log.Println("Building forest...")
	samples := RandomData(NumSamples)
	forest := idtrees.BuildForest(NumTrees, samples, features, SubSamples, SubAttrs,
		func(s []idtrees.Sample, a []idtrees.Attr) *idtrees.Tree {
			return idtrees.ID3(s, a, 0)
		})
	log.Println("Cross validating...")
	validation := RandomData(10000)
	log.Printf("Baseline: %.02f%%", Baseline(validation))
	log.Printf("Validation score: %s", ClassifierScore(forest, validation))

	log.Println("TODO: save to file.")
}

func Baseline(data []idtrees.Sample) float64 {
	countMap := map[idtrees.Class]int{}
	for _, x := range data {
		countMap[x.Class()]++
	}
	var best float64
	for _, count := range countMap {
		score := float64(count) / float64(len(data))
		best = math.Max(best, score)
	}
	return best * 100
}

func ClassifierScore(f idtrees.Forest, data []idtrees.Sample) string {
	rightCounts := map[int]int{}
	totalCounts := map[int]int{}

	var numRight int
	var numTotal int
	for _, test := range data {
		guess := ClassifyCube(f, test)
		if guess == test.Class().(int) {
			numRight++
			rightCounts[guess]++
		}
		totalCounts[test.Class().(int)]++
		numTotal++
	}

	res := fmt.Sprintf("total: %.02f%%", 100*float64(numRight)/float64(numTotal))
	for class, total := range totalCounts {
		right := rightCounts[class]
		res += fmt.Sprintf(", %d: %.02f%%", class, 100*float64(right)/float64(total))
	}

	return res
}

func ClassifyCube(f idtrees.Forest, test idtrees.AttrMap) int {
	outputs := f.Classify(test)
	var maxVal float64
	var maxClass int
	for class, val := range outputs {
		if val > maxVal {
			maxVal = val
			maxClass = class.(int)
		}
	}
	return maxClass
}
