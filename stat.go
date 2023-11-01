// Package stat provides functions for statistical analysis.
package stat

import (
	"math"

	"github.com/aclements/go-moremath/stats"
	"gonum.org/v1/gonum/stat/distuv"
)

// Holm performs the Holm-Bonferroni step-down procedure.
func Holm[S ~[]E, E any](pValues S, get func(E) float64, alpha float64) S {
	m := len(pValues)
	for i, p := range pValues {
		if get(p)*float64(m-i) > alpha {
			return pValues[:i]
		}
	}
	return pValues
}

// Welch performs the Welch t-test.
// It is ported from the python function pingouin.welch_anova.
func Welch(samples [][]float64) (float64, float64) {
	r := len(samples)
	ddof1 := r - 1
	// log.Printf("ddof1 %d", ddof1)

	var totalN, totalMean float64
	means := make([]float64, 0, r)
	weights := make([]float64, 0, r)
	var adjGrandMean, weightsSum float64
	for _, sample := range samples {
		var mean float64
		for j, s := range sample {
			totalN++
			totalMean += (s - totalMean) / totalN
			mean += (s - mean) / float64(j+1)
		}
		means = append(means, mean)

		w := float64(len(sample)) / stats.Variance(sample)
		weights = append(weights, w)

		adjGrandMean += w * mean
		weightsSum += w
	}
	adjGrandMean /= weightsSum
	// log.Printf("weights: %v, adjGrandMean: %f", weights, adjGrandMean)

	var ssRes, ssBet, ssBetAdj float64
	for i, sample := range samples {
		for _, s := range sample {
			d := s - means[i]
			ssRes += d * d
		}

		d := means[i] - totalMean
		ssBet += d * d * float64(len(sample))

		d = means[i] - adjGrandMean
		ssBetAdj += d * d * weights[i]
	}
	msBetAdj := ssBetAdj / float64(ddof1)
	// log.Printf("ssRes: %f, ssBet: %f, ssBetAdj: %f, msBetAdj: %f", ssRes, ssBet, ssBetAdj, msBetAdj)

	var lamb float64
	for i, sample := range samples {
		cnt := 1 / float64(len(sample)-1)
		mw := 1 - weights[i]/weightsSum
		lamb += cnt * mw * mw
	}
	// log.Printf("lamb0: %f", lamb)
	lamb = 3 * lamb / float64(r*r-1)
	fval := msBetAdj / (1 + (2*lamb*(float64(r)-2))/3)
	// log.Printf("lamb: %f, fval: %f", lamb, fval)
	pval := sf(fval, float64(ddof1), 1/lamb)

	return fval, pval
}

// BrownForsythe performs the Brown-Forsythe test of equal variance.
// The input samples are assumed to be already sorted in ascending order.
// It is ported from the python function scipy.stats.levene.
func BrownForsythe(samples [][]float64) (float64, float64) {
	k := len(samples)
	ni := make([]float64, 0, k)
	yci := make([]float64, 0, k)
	var ntot float64
	for j := 0; j < k; j++ {
		nij := float64(len(samples[j]))
		ni = append(ni, nij)
		yci = append(yci, QuantileF(samples[j], 0.5))
		ntot += nij
	}
	// log.Printf("k: %d, ni: %v, yci: %v, ntot: %f", k, ni, yci, ntot)

	zij := make([][]float64, 0, k)
	for i, sample := range samples {
		ycii := yci[i]

		zi := make([]float64, 0, len(sample))
		for _, s := range sample {
			zi = append(zi, math.Abs(s-ycii))
		}
		zij = append(zij, zi)
	}

	zbari := make([]float64, 0, k)
	var zbar float64
	for i, ziji := range zij {
		zbarii := stats.Mean(ziji)
		zbari = append(zbari, zbarii)
		zbar += zbarii * ni[i]
	}
	zbar /= ntot

	var sumz float64
	for i, zbarii := range zbari {
		d := (zbarii - zbar)
		sumz += ni[i] * d * d
	}
	numer := (ntot - float64(k)) * sumz
	// log.Printf("zbari: %v, zbar: %f, numer: %f", zbari, zbar, numer)

	var dvar float64
	for i, ziji := range zij {
		zbarii := zbari[i]

		for _, zijij := range ziji {
			d := zijij - zbarii
			dvar += d * d
		}
	}
	denom := (float64(k) - 1) * dvar
	// log.Printf("dvar: %f, denom: %f", dvar, denom)

	w := numer / denom
	pval := sf(w, float64(k)-1, ntot-float64(k))
	return w, pval
}

func sf(x, dfn, dfd float64) float64 {
	f := distuv.F{D1: dfn, D2: dfd}
	return f.Survival(x)
}

// Mean returns the mean of a sample
func Mean[S ~[]E, E any](sample S, get func(E) float64) float64 {
	var m float64
	for i, d := range sample {
		m += (get(d) - m) / float64(i+1)
	}
	return m
}

// SS returns the mean and sum of squares of a sample.
func SS[S ~[]E, E any](sample S, get func(E) float64) (float64, float64) {
	mean, M2 := 0.0, 0.0
	for n, d := range sample {
		x := get(d)
		delta := x - mean
		mean += delta / float64(n+1)
		M2 += delta * (x - mean)
	}
	return mean, M2
}

// Quantile returns the q-quantile of a sample.
// The sample is assumed to be already sorted.
// It uses the R8 method described in https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm.
func Quantile[S ~[]E, E any](sample S, get func(E) float64, q float64) float64 {
	N := float64(len(sample))
	n := 1/3.0 + q*(N+1/3.0)

	kf, frac := math.Modf(n)
	k := int(kf)
	if k <= 0 {
		return get(sample[0])
	} else if k >= len(sample) {
		return get(sample[len(sample)-1])
	}

	return get(sample[k-1]) + frac*(get(sample[k])-get(sample[k-1]))
}

// QuantileF returns the q-quantile of a sample of float64s.
// The sample is assumed to be already sorted.
func QuantileF(sample []float64, q float64) float64 {
	return Quantile(sample, func(f float64) float64 { return f }, q)
}
