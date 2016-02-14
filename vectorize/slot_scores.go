package vectorize

import "github.com/unixpickle/gocube"

// SlotScores generates a vector where each entry corresponds to a pair of pieces on the cube.
// The "closer" to being solved each pair of pieces is, the higher the entry's value.
func SlotScores(cube gocube.CubieCube) []float64 {
	res := make([]float64, 0, 12*8+12*11+8*7)

	for edge := 0; edge < 12; edge++ {
		for corner := 0; corner < 8; corner++ {
			score := 1.0
			if cube.Edges[edge].Piece == edge {
				score *= 12
			}
			if cube.Edges[edge].Flip == false {
				score *= 2
			}
			if cube.Corners[corner].Piece == corner {
				score *= 8
			}
			if cube.Corners[corner].Orientation == 1 {
				score *= 3
			}
			res = append(res, score)
		}
	}

	for corner := 0; corner < 8; corner++ {
		for corner1 := 0; corner1 < 8; corner1++ {
			if corner == corner1 {
				continue
			}
			score := 1.0
			if cube.Corners[corner].Piece == corner {
				score *= 8
			}
			if cube.Corners[corner].Orientation == 1 {
				score *= 3
			}
			if cube.Corners[corner1].Piece == corner1 {
				score *= 8
			}
			if cube.Corners[corner1].Orientation == 1 {
				score *= 3
			}
			res = append(res, score)
		}
	}

	for edge := 0; edge < 12; edge++ {
		for edge1 := 0; edge1 < 12; edge1++ {
			if edge == edge1 {
				continue
			}
			score := 1.0
			if cube.Edges[edge].Piece == edge {
				score *= 12
			}
			if cube.Edges[edge].Flip == false {
				score *= 2
			}
			if cube.Edges[edge1].Piece == edge {
				score *= 12
			}
			if cube.Edges[edge1].Flip == false {
				score *= 2
			}
			res = append(res, score)
		}
	}

	return res
}
