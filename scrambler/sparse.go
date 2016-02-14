package scrambler

import (
	"math/rand"

	"github.com/unixpickle/gocube"
)

const sparseExhaustiveDepth = 5

// Sparse generates some (but not necessarily all) scrambles up to a given move count.
// The width argument specifies the maximum number of scrambles to generate for each move count.
func Sparse(maxMoves, width int) map[gocube.CubieCube]int {
	if maxMoves <= sparseExhaustiveDepth {
		res := Exhaustive(maxMoves)
		return pruneScrambles(res, width)
	}

	exhaustivePart := Exhaustive(sparseExhaustiveDepth)
	sparsePart := map[gocube.CubieCube]int{}

	for moveCount := sparseExhaustiveDepth + 1; moveCount <= maxMoves; moveCount++ {
		count := 0
		for count < width {
			random := randomMoves(moveCount)
			if _, ok := sparsePart[random]; ok {
				continue
			}
			if computeMoveCount(random, exhaustivePart) != moveCount {
				continue
			}
			sparsePart[random] = moveCount
			count++
		}
	}

	for cube, depth := range sparsePart {
		exhaustivePart[cube] = depth
	}
	return pruneScrambles(exhaustivePart, width)
}

// pruneScrambles removes scrambles to ensure that there are no more than width scrambles per move
// count.
func pruneScrambles(scrambles map[gocube.CubieCube]int, width int) map[gocube.CubieCube]int {
	cubesForCount := map[int][]gocube.CubieCube{}
	for cube, depth := range scrambles {
		if _, ok := cubesForCount[depth]; !ok {
			cubesForCount[depth] = []gocube.CubieCube{}
		}
		cubesForCount[depth] = append(cubesForCount[depth], cube)
	}

	for key, list := range cubesForCount {
		perm := rand.Perm(len(list))
		newList := make([]gocube.CubieCube, len(list))
		for i := range newList {
			newList[i] = list[perm[i]]
		}
		cubesForCount[key] = newList
	}

	res := map[gocube.CubieCube]int{}
	for depth, list := range cubesForCount {
		length := len(list)
		if length > width {
			length = width
		}
		for _, cube := range list[0:length] {
			res[cube] = depth
		}
	}

	return res
}

// randomMoves generates a cube by applying entirely random moves to the identity.
func randomMoves(moveCount int) gocube.CubieCube {
	res := gocube.SolvedCubieCube()
	var lastMove gocube.Move
	for i := 0; i < moveCount; i++ {
		randMove := gocube.Move(rand.Intn(18))
		for i != 0 && randMove.Face() == lastMove.Face() {
			randMove = gocube.Move(rand.Intn(18))
		}
		res.Move(randMove)
		lastMove = randMove
	}
	return res
}

// computeMoveCount computes the move count for a cube.
// You must supply this with a map of cubes from Exhaustive(), so that the leaves of the map are
// exhaustive and all of the same depth.
func computeMoveCount(cube gocube.CubieCube, table map[gocube.CubieCube]int) int {
	for depth := 0; depth < 20; depth++ {
		moveCount := search(cube, table, depth, -1)
		if moveCount >= 0 {
			return moveCount
		}
	}
	panic("no solution found")
}

func search(cube gocube.CubieCube, table map[gocube.CubieCube]int, remainingMoves int,
	lastFace int) int {
	if remainingMoves == 0 {
		if depth, ok := table[cube]; ok {
			return depth
		}
		return -1
	}

	for i := 0; i < 18; i++ {
		move := gocube.Move(i)
		if move.Face() == lastFace {
			continue
		}
		newCube := cube
		newCube.Move(move)
		if res := search(newCube, table, remainingMoves-1, move.Face()); res >= 0 {
			return res + 1
		}
	}

	return -1
}
