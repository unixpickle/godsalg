package scrambler

import "github.com/unixpickle/gocube"

type generationNode struct {
	cube     gocube.CubieCube
	depth    int
	lastFace int
}

// Exhaustive generates every scramble up to a given move count.
func Exhaustive(maxMoves int) map[gocube.CubieCube]int {
	res := map[gocube.CubieCube]int{
		gocube.SolvedCubieCube(): 0,
	}
	nodes := []generationNode{{cube: gocube.SolvedCubieCube(), depth: 0, lastFace: -1}}

	for len(nodes) > 0 {
		next := nodes[0]
		nodes = nodes[1:]
		for i := 0; i < 18; i++ {
			move := gocube.Move(i)
			if move.Face() == next.lastFace {
				continue
			}
			cube := next.cube
			cube.Move(move)
			if _, ok := res[cube]; !ok {
				res[cube] = next.depth + 1
				if next.depth+1 < maxMoves {
					nodes = append(nodes, generationNode{
						cube:     cube,
						depth:    next.depth + 1,
						lastFace: move.Face(),
					})
				}
			}
		}
	}

	return res
}
