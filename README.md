# Utility Functions for Optimization

Seminar report and code for **Mathematical Concepts of Machine Learning**,
RWTH Aachen University. The work is based on Chapter 6 of Roman Garnett's
*Bayesian Optimization* (Cambridge University Press, 2023), and extends it
with a numerical case study.

Bayesian optimization is usually presented as a catalogue of acquisition
functions. This report works through the construction underneath them: each
one is a single step of lookahead on a *utility function* that scores the
collected data. The model says what we believe; the utility says what we
want; only the pair determines an action.

The case study is a campaign resource allocation problem. Electoral
districts are placed on a one-dimensional demographic axis, a Gaussian
process models the vote-margin shift that campaigning in each district would
produce, and polling is the expensive noisy observation. Six utility
functions are then computed on the same data and compared.

## Contents

| File | Description |
|---|---|
| `Seminar Report.pdf` | The report (7 pages, IEEE conference format) |
| `Seminar Slides.pdf` | Presentation slides |
| `gp_core.py` | Gaussian process, the six utility functionals, three acquisition functions |
| `experiments.py` | The four experiments and all figure generation |

## Findings

**Utilities disagree, and the disagreement is large.** On two datasets the
six utilities split four against two. Dataset A has simple reward $-0.98$
and global reward $+1.52$ — a swing of 2.5 points produced by nothing but
the choice of action space.

**Information gain about $x^*$ and about $f^*$ can move in opposite
directions.** Dataset A is far more informative about *where* the optimum
lies (0.878 against 0.033 nats) but *less* informative about its value
(0.004 against 0.062 nats), because the posterior standard deviation at the
implied peak is still 2.08 points. The source treats the two as moving
together; in this instance they do not.

**`max y` is a safe proxy for simple reward only above SNR ≈ 3.** Below that
it ranks measurement luck: at unit signal-to-noise, roughly one dataset
comparison in nine comes out backwards.

**The utility decides whether a policy comparison is measurable at all.**
Across 120 identical runs of EI, KG, GP-UCB and random search, resolving the
gap between the top two policies needs about 3 replicates when scored by
information gain about $x^*$ and about 390 when scored by global reward.
Expected Improvement learns no more about the optimum's location than
uniform random sampling does ($-0.07 \pm 0.06$ nats, paired).

## Reproducing

```bash
pip install numpy scipy matplotlib
python experiments.py
```

Writes `fig_datasets.pdf`, `fig_maxy.pdf`, `fig_risk.pdf`, `fig_policy.pdf`
and `results.json`. Experiments 1–3 finish in a few minutes; experiment 4
runs 120 replicates and takes roughly 15 minutes, printing progress as it
goes. All seeds are fixed, so the numbers in the report reproduce exactly.

## Implementation notes

Everything is exact Gaussian process conditioning in NumPy — no GP library —
so the definitions used are visibly the ones in the report. The prior is
zero-mean with a squared exponential covariance, amplitude $\sigma_f = 3$
points, length scale $\ell = 0.08$, observation noise $\sigma_n = 0.5$, on a
uniform grid of 501 districts.

Information gain is estimated by Monte Carlo over posterior sample paths:
$H[x^*]$ as a discrete entropy over grid cells, $H[f^*]$ from the sampled
maxima via the Vasicek spacing estimator. Because information gain is a
difference of entropies on the same grid, the discretization constant
cancels.

The Knowledge Gradient uses the exact expectation of a maximum of affine
functions of a standard normal, via the epigraph algorithm of Frazier,
Powell and Dayanık (2009). It was checked against a 3,000,000-sample Monte
Carlo estimate on 30 random instances; worst discrepancy 1.7 standard
errors, consistent with Monte Carlo noise alone.

## References

- R. Garnett, *Bayesian Optimization*. Cambridge University Press, 2023.
- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. MIT Press, 2006.
- D. R. Jones, M. Schonlau and W. J. Welch, "Efficient global optimization of expensive black-box functions," *J. Global Optim.*, vol. 13, no. 4, pp. 455–492, 1998.
- P. I. Frazier, W. B. Powell and S. Dayanık, "The knowledge-gradient policy for correlated normal beliefs," *INFORMS J. Comput.*, vol. 21, no. 4, pp. 599–613, 2009.

## Tooling

The Gaussian process implementation, the experiments and the figures in this
repository were developed with the assistance of a large language model, as
was part of the report text. The topic, the campaign case study and the
direction of the work are my own.

## Author

Fahimeh Fereydounian — Faculty of Computer Science, RWTH Aachen University
