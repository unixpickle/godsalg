package main

import (
	"math/rand"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/weakai/idtrees"
)

type DataPoint struct {
	Stickers *gocube.StickerCube
	Cubies   *gocube.CubieCube
	Moves    int
}

func RandomData(count int) []idtrees.Sample {
	var res []idtrees.Sample
	for i := 0; i < count; i++ {
		moves := rand.Intn(MaxMoves-MinMoves+1) + MinMoves
		cubies := RandomScramble(moves)
		stickers := cubies.StickerCube()
		res = append(res, &DataPoint{
			Cubies:   cubies,
			Stickers: &stickers,
			Moves:    moves,
		})
	}
	return res
}

func (d *DataPoint) Attr(a idtrees.Attr) idtrees.Val {
	idx := a.(int)
	if idx < 54 {
		return d.Stickers[idx]
	} else if idx < 54+12 {
		return d.Cubies.Edges[idx-54].Piece
	} else {
		return d.Cubies.Corners[idx-(54+12)].Piece
	}
}

func (d *DataPoint) Class() idtrees.Class {
	if d.Moves > 11 {
		return 1
	} else {
		return 0
	}
}
