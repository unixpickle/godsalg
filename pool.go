package main

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	SubSamples = 10000
	SubAttrs   = 9
)

type Classifier struct {
	Tree *idtrees.Tree
}

func (c *Classifier) Classify(s boosting.SampleList) linalg.Vector {
	return classificationVec(c.Tree, s.(SampleList))
}

type Pool struct {
	Attrs []idtrees.Attr
}

func (p *Pool) BestClassifier(s boosting.SampleList, weights linalg.Vector) boosting.Classifier {
	var meanWeight float64
	for _, w := range weights {
		meanWeight += math.Pow(w, 2)
	}
	meanWeight /= float64(s.Len())

	chooseProb := float64(SubSamples) / float64(s.Len())
	var samples []idtrees.Sample
	for i, w := range weights {
		desirability := chooseProb * math.Pow(w, 2) / meanWeight
		if math.Abs(rand.NormFloat64()) < desirability {
			samples = append(samples, s.(SampleList)[i])
		}
	}

	attrs := make([]idtrees.Attr, SubAttrs)
	perm := rand.Perm(len(p.Attrs))
	for i, x := range perm[:SubAttrs] {
		attrs[i] = p.Attrs[x]
	}

	tree := idtrees.ID3(samples, attrs, 0)
	return &Classifier{
		Tree: tree,
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

type weightSorter struct {
	weights linalg.Vector
	indices []int
}

func (w *weightSorter) Len() int {
	return len(w.weights)
}

func (w *weightSorter) Swap(i, j int) {
	w.weights[i], w.weights[j] = w.weights[j], w.weights[i]
	w.indices[i], w.indices[j] = w.indices[j], w.indices[i]
}

func (w *weightSorter) Less(i, j int) bool {
	return w.weights[i] > w.weights[j]
}
