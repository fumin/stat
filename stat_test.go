package stat

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os/exec"
	"slices"
	"strconv"
	"testing"

	"github.com/pkg/errors"
)

func TestMean(t *testing.T) {
	sample := []float64{0.4, 3.6, 4.5, 5.0, 7.5}
	expected := 4.2
	mean := Mean(sample, func(f float64) float64 { return f })
	if mean != expected {
		t.Fatalf("%f %f", mean, expected)
	}
}

func TestSS(t *testing.T) {
	sample := []float64{0.6, 0.47, 0.17, 0.43, 0.3}
	expectedMean, expectedSS := 0.394, 0.10852
	mean, ss := SS(sample, func(f float64) float64 { return f })
	if math.Abs(mean-expectedMean) > 1e-6 {
		t.Fatalf("%f %f", mean, expectedMean)
	}
	if math.Abs(ss-expectedSS) > 1e-6 {
		t.Fatalf("%f %f", ss, expectedSS)
	}
}

type pythonIn struct {
	Func    string
	Samples [][]float64
}

type pythonOut struct {
	BrownForsythe struct {
		Statistic float64
		PValue    float64
	}
	Welch struct {
		F      float64
		PValue float64
	}
	Holm struct {
		Reject           []bool
		PValuesCorrected []float64
	}
}

func TestHolm(t *testing.T) {
	type testcase struct {
		name string
		in   pythonIn
	}
	cases := make([]testcase, 0, 100)
	for i := 0; i < cap(cases); i++ {
		n := rand.Intn(100) + 1
		pValues := make([]float64, 0, n)
		for j := 0; j < n; j++ {
			pValues = append(pValues, 0.1/float64(n)*rand.Float64())
		}
		tc := testcase{name: strconv.Itoa(i), in: pythonIn{Func: "Holm", Samples: [][]float64{pValues}}}
		cases = append(cases, tc)
	}
	// cases = []testcase{{name: "fixed", in: pythonIn{Func: "Holm", Samples: [][]float64{{0.50, 0.003, 0.32, 0.054, 0.0003}}}}}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			out, b, err := python(tc.in)
			if err != nil {
				t.Fatalf("%+v %#v", err, tc)
			}
			rejectsExpected := make([]float64, 0)
			pvals := tc.in.Samples[0]
			for i, r := range out.Holm.Reject {
				if r {
					rejectsExpected = append(rejectsExpected, pvals[i])
				}
			}
			slices.Sort(rejectsExpected)

			slices.Sort(pvals)
			rejects := Holm(pvals, func(f float64) float64 { return f }, 0.05)
			if !slices.Equal(rejects, rejectsExpected) {
				t.Fatalf("%#v %#v %#v %s", rejects, tc, out, b)
			}
		})
	}
}

func TestBrownForsythe(t *testing.T) {
	type testcase struct {
		name string
		in   pythonIn
	}
	cases := make([]testcase, 0)
	for _, fn := range []string{"BrownForsythe", "Welch"} {
		for i := 0; i < 100; i++ {
			m := rand.Intn(98) + 2
			samples := make([][]float64, 0, m)
			for j := 0; j < m; j++ {
				n := rand.Intn(98) + 2
				s := make([]float64, 0, n)
				for k := 0; k < n; k++ {
					f := 100 * (2*rand.Float64() - 1)
					s = append(s, f)
				}
				samples = append(samples, s)
			}
			cases = append(cases, testcase{name: fmt.Sprintf("%s%d", fn, i), in: pythonIn{Func: fn, Samples: samples}})
		}
	}
	// cases = []testcase{{name: "fixed", in: pythonIn{Func: "Welch", Samples: [][]float64{
	// 	{7, 14, 14, 13, 12, 9, 6, 14, 12, 8},
	// 	{15, 17, 13, 15, 15, 13, 9, 12, 10, 8},
	// 	{6, 8, 8, 9, 5, 14, 13, 8, 10, 9},
	// }}}}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			out, b, err := python(tc.in)
			if err != nil {
				t.Fatalf("%+v %#v", err, tc)
			}

			var statistic, pValue float64
			var statisticExpected, pValueExpected float64
			switch tc.in.Func {
			case "BrownForsythe":
				for _, sample := range tc.in.Samples {
					slices.Sort(sample)
				}
				statistic, pValue = BrownForsythe(tc.in.Samples)
				statisticExpected, pValueExpected = out.BrownForsythe.Statistic, out.BrownForsythe.PValue
			case "Welch":
				statistic, pValue = Welch(tc.in.Samples)
				statisticExpected, pValueExpected = out.Welch.F, out.Welch.PValue
			default:
				t.Fatalf("%#v", tc)
			}

			if math.Abs(statistic/statisticExpected-1) > 1e-6 {
				t.Fatalf("%f %#v %#v %s", statistic, out, tc, b)
			}
			if pValueExpected < 1e-6 {
				if pValue > 1e-6 {
					t.Fatalf("%f %#v %#v %s", pValue, out, tc, b)
				}
			} else {
				if math.Abs(pValue/pValueExpected-1) > 1e-6 {
					t.Fatalf("%f %#v %#v %s", pValue, out, tc, b)
				}
			}
		})
	}
}

func python(in pythonIn) (pythonOut, []byte, error) {
	argv1, err := json.Marshal(in)
	if err != nil {
		return pythonOut{}, nil, errors.Wrap(err, "")
	}
	oe, err := exec.Command("python3", "stat.py", string(argv1)).CombinedOutput()
	if err != nil {
		return pythonOut{}, nil, errors.Wrap(err, fmt.Sprintf("%s", oe))
	}

	var lastLine string
	scanner := bufio.NewScanner(bytes.NewBuffer(oe))
	for scanner.Scan() {
		lastLine = scanner.Text()
	}
	if err := scanner.Err(); err != nil {
		return pythonOut{}, nil, errors.Wrap(err, fmt.Sprintf("%s", oe))
	}

	var out pythonOut
	if err := json.Unmarshal([]byte(lastLine), &out); err != nil {
		return pythonOut{}, nil, errors.Wrap(err, fmt.Sprintf("%s", oe))
	}
	return out, oe, nil
}
