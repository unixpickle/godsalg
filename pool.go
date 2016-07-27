package main

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	SubSamples = 4000
	SubAttrs   = 30
)

type Classifier struct {
	Tree    *idtrees.Tree
	Outputs linalg.Vector
}

func (c *Classifier) Classify(s boosting.SampleList) linalg.Vector {
	if c.Outputs == nil {
		return classificationVec(c.Tree, s.(SampleList))
	} else {
		res := make(linalg.Vector, len(c.Outputs))
		copy(res, c.Outputs)
		return res
	}
}

type Pool struct {
	Trees   []*idtrees.Tree
	Outputs blas64.General
}

func NewPool(n int, samples []idtrees.Sample, attrs []idtrees.Attr) *Pool {
	var res Pool

	res.Outputs.Stride = len(samples)
	res.Outputs.Cols = len(samples)

	forest := idtrees.BuildForest(n, samples, attrs, SubSamples, SubAttrs,
		func(s []idtrees.Sample, a []idtrees.Attr) *idtrees.Tree {
			return idtrees.ID3(s, a, 0)
		})

	for _, tree := range forest {
		v := classificationVec(tree, samples)
		res.Outputs.Data = append(res.Outputs.Data, v...)
		res.Outputs.Rows++
		res.Trees = append(res.Trees, tree)
	}

	return &res
}

func (p *Pool) BestClassifier(s boosting.SampleList, weights linalg.Vector) boosting.Classifier {
	vec := blas64.Vector{
		Inc:  1,
		Data: weights,
	}
	output := blas64.Vector{
		Inc:  1,
		Data: make([]float64, len(p.Trees)),
	}
	blas64.Gemv(blas.NoTrans, 1, p.Outputs, vec, 0, output)
	largest := blas64.Iamax(len(p.Trees), output)
	startIdx := largest * p.Outputs.Cols
	return &Classifier{
		Tree:    p.Trees[largest],
		Outputs: p.Outputs.Data[startIdx : startIdx+p.Outputs.Cols],
	}
}

func classificationVec(t *idtrees.Tree, s []idtrees.Sample) linalg.Vector {
	res := make(linalg.Vector, len(s))
	for i, sample := range s {
		if classify(t, sample) == 0 {
			res[i] = -1
		} else {
			res[i] = 1
		}
	}
	return res
}

func classify(t *idtrees.Tree, sample idtrees.AttrMap) int {
	outputs := t.Classify(sample)
	var maxVal float64
	var maxClass int
	for class, val := range outputs {
		if val > maxVal {
			maxVal = val
			maxClass = class.(int)
		}
	}
	return maxClass
}
