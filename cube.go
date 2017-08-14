package godsalg

import (
	"math"
	"math/rand"

	"github.com/unixpickle/gocube"
)

// RandomScramble generates a move-based scramble
// of a certain length.
// It returns the inverse of the last move, which doubles
// as the first move of a valid solution.
func RandomScramble(length int) (*gocube.CubieCube, gocube.Move) {
	moves := allMoves()
	res := gocube.SolvedCubieCube()
	axis := -1
	lastMove := gocube.Move(0)
	for i := 0; i < length; i++ {
		move := moves[rand.Intn(len(moves))]
		if moveAxis(move) != axis {
			moves = allMoves()
			axis = moveAxis(move)
		}
		for i := 0; i < len(moves); i++ {
			if moves[i].Face() == move.Face() {
				moves[i] = moves[len(moves)-1]
				moves = moves[:len(moves)-1]
				i--
			}
		}
		res.Move(move)
		lastMove = move
	}
	return &res, lastMove.Inverse()
}

// CubeVector returns a vectorized representation of
// the stickers of a cube.
func CubeVector(c *gocube.CubieCube) []float64 {
	stickerCube := c.StickerCube()
	res := make([]float64, 8*6*6)

	mean := 1.0 / 6
	stddev := math.Sqrt(0.13937)

	var stickerIdx int
	for i, sticker := range stickerCube[:] {
		if i%9 == 4 {
			continue
		}
		for j := 0; j < 6; j++ {
			if j == sticker-1 {
				res[j+stickerIdx] = (1 - mean) / stddev
			} else {
				res[j+stickerIdx] = (0 - mean) / stddev
			}
		}
		stickerIdx += 6
	}

	return res
}

func allMoves() []gocube.Move {
	res := make([]gocube.Move, NumMoves)
	for i := range res {
		res[i] = gocube.Move(i)
	}
	return res
}

func moveAxis(m gocube.Move) int {
	return (m.Face() - 1) / 2
}
