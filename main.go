package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"

	"github.com/unixpickle/gocube"
)

const MaxFullSearchDepth = 5
const MaxCountPerDepth = 10000

type SearchNode struct {
	cube  gocube.CubieCube
	depth int
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: godsalg <output.csv>")
		os.Exit(1)
	}

	nodes := []SearchNode{SearchNode{gocube.SolvedCubieCube(), 0}}
	visited := map[gocube.CubieCube]int{}
	for len(nodes) > 0 {
		node := nodes[0]
		nodes = nodes[1:]
		if _, ok := visited[node.cube]; !ok {
			visited[node.cube] = node.depth
			if node.depth == MaxFullSearchDepth {
				continue
			}
			for moveIdx := 0; moveIdx < 18; moveIdx++ {
				cube := node.cube
				cube.Move(gocube.Move(moveIdx))
				nodes = append(nodes, SearchNode{cube, node.depth + 1})
			}
		}
	}

	var buf bytes.Buffer
	buf.WriteString("*Move Count")
	for i := 0; i < 8; i++ {
		buf.WriteRune(',')
		buf.WriteString("CornerPiece")
		buf.WriteString(strconv.Itoa(i))
		buf.WriteRune(',')
		buf.WriteString("CornerOrientation")
		buf.WriteString(strconv.Itoa(i))
		buf.WriteRune(',')
		buf.WriteString("Corner")
		buf.WriteString(strconv.Itoa(i))
	}
	for i := 0; i < 12; i++ {
		buf.WriteRune(',')
		buf.WriteString("EdgePiece")
		buf.WriteString(strconv.Itoa(i))
		buf.WriteRune(',')
		buf.WriteString("EdgeOrientation")
		buf.WriteString(strconv.Itoa(i))
		buf.WriteRune(',')
		buf.WriteString("Edge")
		buf.WriteString(strconv.Itoa(i))
	}
	for i := 0; i < 12; i++ {
		for j := 0; j < 8; j++ {
			buf.WriteString(",EdgeCorner")
			buf.WriteString(strconv.Itoa(i))
			buf.WriteRune('-')
			buf.WriteString(strconv.Itoa(j))
		}
	}
	for i := 0; i < 12; i++ {
		for j := 0; j < 8; j++ {
			buf.WriteString(",EOCO")
			buf.WriteString(strconv.Itoa(i))
			buf.WriteRune('-')
			buf.WriteString(strconv.Itoa(j))
		}
	}

	numPerDepth := map[int]int{}
	for cube, depth := range visited {
		if numPerDepth[depth] == MaxCountPerDepth {
			continue
		}
		numPerDepth[depth]++
		buf.WriteRune('\n')
		csvRowForCube(&buf, depth, cube)
	}

	if err := ioutil.WriteFile(os.Args[1], buf.Bytes(), 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func csvRowForCube(buf *bytes.Buffer, depth int, c gocube.CubieCube) {
	buf.WriteString(strconv.Itoa(depth))
	for i := 0; i < 8; i++ {
		buf.WriteString(",_")
		buf.WriteString(strconv.Itoa(c.Corners[i].Piece))
		buf.WriteRune(',')
		buf.WriteString(orientationAxis(c.Corners[i].Orientation))
		buf.WriteString(",")
		buf.WriteString(strconv.Itoa(c.Corners[i].Piece))
		buf.WriteRune('-')
		buf.WriteString(orientationAxis(c.Corners[i].Orientation))
	}
	for i := 0; i < 12; i++ {
		buf.WriteString(",_")
		buf.WriteString(strconv.Itoa(c.Edges[i].Piece))
		buf.WriteRune(',')
		if c.Edges[i].Flip {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
		buf.WriteString(",")
		buf.WriteString(strconv.Itoa(c.Edges[i].Piece))
		buf.WriteRune('-')
		if c.Edges[i].Flip {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	}
	for i := 0; i < 12; i++ {
		for j := 0; j < 8; j++ {
			buf.WriteRune(',')
			buf.WriteString(strconv.Itoa(c.Edges[i].Piece))
			buf.WriteRune('-')
			buf.WriteString(strconv.Itoa(c.Corners[j].Piece))
		}
	}
	for i := 0; i < 12; i++ {
		for j := 0; j < 8; j++ {
			buf.WriteRune(',')
			if c.Edges[i].Flip {
				buf.WriteString("true")
			} else {
				buf.WriteString("false")
			}
			buf.WriteRune('-')
			buf.WriteString(orientationAxis(c.Corners[j].Orientation))
		}
	}
}

func orientationAxis(orientation int) string {
	switch orientation {
	case 0:
		return "x"
	case 1:
		return "y"
	case 2:
		return "z"
	}
	panic("unknown orientation")
}
