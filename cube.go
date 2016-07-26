package main

import (
	"math/rand"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/num-analysis/linalg"
)

// RandomScramble generates a move-based scramble
// of a certain length.
func RandomScramble(length int) *gocube.CubieCube {
	moves := allMoves()
	res := gocube.SolvedCubieCube()
	axis := -1
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
	}
	return &res
}

// CubeVector returns a vectorized representation of
// the stickers of a cube.
func CubeVector(c *gocube.CubieCube) linalg.Vector {
	stickerCube := c.StickerCube()
	res := make(linalg.Vector, 8*6*6)

	var stickerIdx int
	for i, sticker := range stickerCube[:] {
		if i%9 == 4 {
			continue
		}
		for j := 0; j < 6; j++ {
			if j == sticker-1 {
				res[j+stickerIdx] = 1.0
			} else {
				res[j+stickerIdx] = -0.2
			}
		}
		stickerIdx += 6
	}

	return res
}

func allMoves() []gocube.Move {
	res := make([]gocube.Move, 18)
	for i := range res {
		res[i] = gocube.Move(i)
	}
	return res
}

func moveAxis(m gocube.Move) int {
	return (m.Face() - 1) / 2
}
