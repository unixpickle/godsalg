package main

import (
	"fmt"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/godsalg/scrambler"
	"github.com/unixpickle/godsalg/vectorize"
	"github.com/unixpickle/weakai/svm"
)

const maxDepth = 7

func main() {
	fmt.Println("Generating cubes...")
	m := moveToCube(maxDepth, 2000)

	fmt.Println("Formulating SVM problem...")

	problem := &svm.Problem{
		Positives: make([]svm.Sample, 0),
		Negatives: make([]svm.Sample, 0),
		Kernel:    svm.LinearKernel,
	}

	for depth, cubes := range m {
		for _, cube := range cubes {
			vec := svm.Sample(vectorize.SlotScores(cube))
			if depth == maxDepth {
				problem.Positives = append(problem.Positives, vec)
			} else {
				problem.Negatives = append(problem.Negatives, vec)
			}
		}
	}

	fmt.Println("Solving...")
	solver := svm.GradientDescentSolver{
		Steps:    10000,
		StepSize: 0.01,
		Tradeoff: 0.001,
	}
	solution := solver.Solve(problem)

	rateClassifier(m, solution)
}

func moveToCube(maxMoves, width int) map[int][]gocube.CubieCube {
	res := map[int][]gocube.CubieCube{}
	for i := 0; i <= maxMoves; i++ {
		res[i] = []gocube.CubieCube{}
	}

	cubes := scrambler.Sparse(maxMoves, width)
	for cube, depth := range cubes {
		res[depth] = append(res[depth], cube)
	}

	return res
}

func rateClassifier(cubes map[int][]gocube.CubieCube, classifier svm.Classifier) {
	for depth := 0; depth <= maxDepth; depth++ {
		depthCubes := cubes[depth]
		var numRight int
		var numWrong int
		for _, cube := range depthCubes {
			class := classifier.Classify(vectorize.SlotScores(cube))
			if class == (depth == maxDepth) {
				numRight++
			} else {
				numWrong++
			}
		}
		fmt.Println("Depth", depth, "got", numRight, "/", (numRight + numWrong), "=",
			100*float64(numRight)/float64(numRight+numWrong), "%")
	}
}
