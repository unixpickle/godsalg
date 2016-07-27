package main

import (
	"fmt"
	"math/rand"

	"github.com/unixpickle/gocube"
	"github.com/unixpickle/weakai/idtrees"
)

const DataFeatureCount = 54 + 8 + 12 + 4

type SampleList []idtrees.Sample

func (s SampleList) Len() int {
	return len(s)
}

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
	switch {
	case idx < 54:
		return d.Stickers[idx]
	case idx < 54+12:
		return d.Cubies.Edges[idx-54].Piece
	case idx < 54+12+8:
		return d.Cubies.Corners[idx-(54+12)].Piece
	case idx == 54+12+8:
		return d.numBadEdges()
	case idx == 54+12+8+1:
		return d.numOrientedCorners()
	case idx == 54+12+8+2:
		return d.numPositionedEdges()
	case idx == 54+12+8+3:
		return d.numPositionedCorners()
	default:
		panic(fmt.Sprintf("unknown feature: %d", idx))
	}
}

func (d *DataPoint) Class() idtrees.Class {
	if d.Moves > 11 {
		return 1
	} else {
		return 0
	}
}

func (d *DataPoint) numBadEdges() int {
	var count int
	for _, x := range d.Cubies.Edges[:] {
		if x.Flip {
			count++
		}
	}
	return count
}

func (d *DataPoint) numOrientedCorners() int {
	var count int
	for _, x := range d.Cubies.Corners[:] {
		if x.Orientation == 0 {
			count++
		}
	}
	return count
}

func (d *DataPoint) numPositionedEdges() int {
	var count int
	for i, x := range d.Cubies.Edges[:] {
		if x.Piece == i {
			count++
		}
	}
	return count
}

func (d *DataPoint) numPositionedCorners() int {
	var count int
	for i, x := range d.Cubies.Corners[:] {
		if x.Piece == i {
			count++
		}
	}
	return count
}
