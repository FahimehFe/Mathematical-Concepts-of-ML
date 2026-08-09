"""
Core Gaussian process machinery and utility functionals for the campaign
resource-allocation case study.

Implements the utility functions of Garnett (2023), Chapter 6:
    simple reward      (6.3)/(6.4)
    global reward      (6.5)
    max-y              (6.6)   [the "nonsensical alternative"]
    cumulative reward  (6.7)
    information gain   (6.8)   about x* and about f*

Everything is exact GP conditioning with a squared-exponential kernel; no
external GP library is used so that the paper is fully self-contained.
"""

import numpy as np
from scipy.stats import norm, differential_entropy

# ----------------------------------------------------------------------------
# Gaussian process
# ----------------------------------------------------------------------------


class GP:
    """Zero-mean GP with squared-exponential covariance and iid Gaussian noise."""

    def __init__(self, lengthscale=0.08, amplitude=1.0, noise=0.10, mean=0.0):
        self.ell = lengthscale
        self.sf = amplitude
        self.sn = noise
        self.m = mean

    def k(self, a, b):
        a = np.atleast_1d(a)[:, None]
        b = np.atleast_1d(b)[None, :]
        return self.sf**2 * np.exp(-0.5 * ((a - b) / self.ell) ** 2)

    def posterior(self, X, y, Xs):
        """Return posterior mean and full covariance at Xs given data (X, y)."""
        X = np.atleast_1d(X)
        y = np.atleast_1d(y)
        if X.size == 0:
            mu = np.full(Xs.shape, self.m)
            return mu, self.k(Xs, Xs)
        K = self.k(X, X) + (self.sn**2) * np.eye(X.size)
        Ks = self.k(X, Xs)
        L = np.linalg.cholesky(K + 1e-10 * np.eye(X.size))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - self.m))
        mu = self.m + Ks.T @ alpha
        v = np.linalg.solve(L, Ks)
        cov = self.k(Xs, Xs) - v.T @ v
        return mu, cov

    def posterior_marginal(self, X, y, Xs):
        mu, cov = self.posterior(X, y, Xs)
        return mu, np.sqrt(np.maximum(np.diag(cov), 1e-12))

    def sample_paths(self, mu, cov, n, rng):
        """Draw n sample paths from N(mu, cov)."""
        d = cov.shape[0]
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
        return mu[None, :] + rng.standard_normal((n, d)) @ L.T


# ----------------------------------------------------------------------------
# Utility functionals  (Garnett 2023, ch. 6)
# ----------------------------------------------------------------------------


def simple_reward(gp, X, y, grid):
    """(6.3) max posterior mean restricted to the visited locations."""
    if np.size(X) == 0:
        return -np.inf
    mu, _ = gp.posterior_marginal(X, y, np.atleast_1d(X))
    return float(np.max(mu))


def global_reward(gp, X, y, grid):
    """(6.5) max posterior mean over the whole domain."""
    mu, _ = gp.posterior_marginal(X, y, grid)
    return float(np.max(mu))


def max_y(gp, X, y, grid):
    """(6.6) maximum raw (noisy) observation -- the flawed alternative."""
    if np.size(y) == 0:
        return -np.inf
    return float(np.max(y))


def cumulative_reward(gp, X, y, grid):
    """(6.7) sum of observed values."""
    if np.size(y) == 0:
        return 0.0
    return float(np.sum(y))


def _xstar_entropy(paths, grid):
    """Discrete entropy (nats) of argmax location over the grid."""
    idx = np.argmax(paths, axis=1)
    counts = np.bincount(idx, minlength=grid.size).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _fstar_entropy(paths):
    """Differential entropy (nats) of the max value, Vasicek spacing estimator."""
    fs = np.max(paths, axis=1)
    return float(differential_entropy(fs))


def information_gain(gp, X, y, grid, rng, n_samples=40000, prior_cache=None):
    """(6.8) entropy reduction about x* and about f*.

    Returns (ig_xstar, ig_fstar) in nats.
    """
    if prior_cache is None:
        pmu, pcov = gp.posterior(np.array([]), np.array([]), grid)
        prior_paths = gp.sample_paths(pmu, pcov, n_samples, rng)
        prior_cache = (_xstar_entropy(prior_paths, grid), _fstar_entropy(prior_paths))
    Hx_prior, Hf_prior = prior_cache

    mu, cov = gp.posterior(X, y, grid)
    paths = gp.sample_paths(mu, cov, n_samples, rng)
    return Hx_prior - _xstar_entropy(paths, grid), Hf_prior - _fstar_entropy(paths)


def prior_entropies(gp, grid, rng, n_samples=40000):
    pmu, pcov = gp.posterior(np.array([]), np.array([]), grid)
    paths = gp.sample_paths(pmu, pcov, n_samples, rng)
    return _xstar_entropy(paths, grid), _fstar_entropy(paths)


def risk_sensitive_recommendation(gp, X, y, grid, beta):
    """argmax over the domain of mu + beta*sigma (sec. 6.1, risk tolerance)."""
    mu, sd = gp.posterior_marginal(X, y, grid)
    return int(np.argmax(mu + beta * sd))


# ----------------------------------------------------------------------------
# Acquisition functions  (one-step lookahead on the utilities above)
# ----------------------------------------------------------------------------


def expected_improvement(gp, X, y, grid):
    """One-step lookahead on simple reward -> EI (Jones et al., 1998)."""
    mu, sd = gp.posterior_marginal(X, y, grid)
    if np.size(X) == 0:
        incumbent = gp.m
    else:
        mu_obs, _ = gp.posterior_marginal(X, y, np.atleast_1d(X))
        incumbent = np.max(mu_obs)
    z = (mu - incumbent) / np.maximum(sd, 1e-12)
    return (mu - incumbent) * norm.cdf(z) + sd * norm.pdf(z)


def _expected_max_affine(a, b):
    """E_z[ max_i (a_i + b_i z) ] for z ~ N(0,1), exactly.

    Frazier's epigraph algorithm for the knowledge-gradient computation.
    """
    order = np.lexsort((a, b))
    a, b = a[order], b[order]

    # among lines with identical slope keep only the highest intercept
    keep = np.ones(a.size, dtype=bool)
    keep[:-1] = b[:-1] != b[1:]
    a, b = a[keep], b[keep]

    if a.size == 1:
        return float(a[0])

    # build the upper envelope
    idx = [0]
    c = [-np.inf, np.inf]
    for i in range(1, a.size):
        while True:
            j = idx[-1]
            cij = (a[j] - a[i]) / (b[i] - b[j])
            if cij <= c[-2]:
                idx.pop()
                c.pop()
                if len(idx) == 0:  # pragma: no cover - defensive
                    idx.append(i)
                    c = [-np.inf, np.inf]
                    break
            else:
                c[-1] = cij
                c.append(np.inf)
                idx.append(i)
                break

    a, b = a[idx], b[idx]
    c = np.array(c)
    Phi = norm.cdf(c)
    phi = norm.pdf(c)
    return float(np.sum(a * (Phi[1:] - Phi[:-1]) + b * (phi[:-1] - phi[1:])))


def knowledge_gradient(gp, X, y, grid):
    """One-step lookahead on global reward -> KG (Frazier et al., 2009)."""
    mu, cov = gp.posterior(X, y, grid)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    current = np.max(mu)
    denom = np.sqrt(sd**2 + gp.sn**2)
    out = np.empty(grid.size)
    for i in range(grid.size):
        btil = cov[:, i] / denom[i]
        out[i] = _expected_max_affine(mu.copy(), btil) - current
    return out


def gp_ucb(gp, X, y, grid, beta=2.0):
    """Cumulative-reward-motivated policy (Srinivas et al., 2010)."""
    mu, sd = gp.posterior_marginal(X, y, grid)
    return mu + beta * sd
