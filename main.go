package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	NumSamples = 300000
	NumTrees   = 1000
	BoostSteps = 2000

	MinMoves = 5
	MaxMoves = 18
)

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output>")
		os.Exit(1)
	}

	log.Println("Building samples...")
	samples := RandomData(NumSamples)

	log.Println("Building trees...")
	var features []idtrees.Attr
	for i := 0; i < DataFeatureCount; i++ {
		features = append(features, i)
	}
	pool := NewPool(NumTrees, samples, features)

	log.Println("Boosting...")
	booster := boosting.Gradient{
		Loss:    boosting.SquareLoss{},
		Desired: ClassVec(samples),
		List:    SampleList(samples),
		Pool:    pool,
	}
	for i := 0; i < BoostSteps; i++ {
		cost := booster.Step()
		log.Println("Epoch", i, "cost", cost)
	}

	log.Println("Cross validating...")
	res := &booster.Sum
	for _, classifier := range res.Classifiers {
		classifier.(*Classifier).Outputs = nil
	}
	validation := RandomData(10000)
	log.Printf("Baseline: %.02f%%", Baseline(validation))
	log.Printf("Validation score: %s", ClassifierScore(res, validation))

	log.Println("TODO: save to file.")
}

func ClassVec(data []idtrees.Sample) linalg.Vector {
	res := make(linalg.Vector, len(data))
	for i, x := range data {
		if x.Class().(int) == 0 {
			res[i] = -1
		} else {
			res[i] = 1
		}
	}
	return res
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

func ClassifierScore(c boosting.Classifier, data []idtrees.Sample) string {
	actualVec := c.Classify(SampleList(data))

	rightCounts := map[int]int{}
	totalCounts := map[int]int{}

	var numRight int
	var numTotal int
	for i, actual := range actualVec {
		realClass := data[i].Class().(int)
		if (realClass == 0) == (actual < 0) {
			numRight++
			rightCounts[realClass]++
		}
		totalCounts[realClass]++
		numTotal++
	}

	res := fmt.Sprintf("total: %.02f%%", 100*float64(numRight)/float64(numTotal))
	for class, total := range totalCounts {
		right := rightCounts[class]
		res += fmt.Sprintf(", %d: %.02f%%", class, 100*float64(right)/float64(total))
	}

	return res
}
