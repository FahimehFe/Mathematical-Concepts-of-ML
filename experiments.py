"""
Experiments for "Utility Functions for Optimization: A Decision-Theoretic
Comparison with a Campaign Resource-Allocation Case Study".

E1  Utilities disagree on the same data          -> fig_datasets.pdf, tab_e1
E2  The max-y pathology and its SNR threshold    -> fig_maxy.pdf,     tab_e2
E3  Risk tolerance costs expected value          -> fig_risk.pdf
E4  Policy / metric mismatch                     -> fig_policy.pdf,   tab_e4
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gp_core import (
    GP,
    simple_reward,
    global_reward,
    max_y,
    cumulative_reward,
    information_gain,
    prior_entropies,
    risk_sensitive_recommendation,
    expected_improvement,
    knowledge_gradient,
    gp_ucb,
)

# --- house style -------------------------------------------------------------
C_BLUE, C_ORANGE, C_GREEN, C_PURPLE = "#0072B2", "#D55E00", "#009E73", "#8B5FBF"
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "grid.color": "#e4e4e4",
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

COL1, COL2 = 3.4, 7.0  # IEEE single- and double-column widths (inches)

RESULTS = {}
GRID = np.linspace(0.0, 1.0, 501)
GP_MODEL = GP(lengthscale=0.08, amplitude=3.0, noise=0.5, mean=0.0)
RNG = np.random.default_rng(20260806)


def band(ax, x, mu, sd, color=C_BLUE):
    ax.fill_between(x, mu - 1.96 * sd, mu + 1.96 * sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, mu, color=color, lw=1.4, zorder=3)


# =============================================================================
# E1 -- the same data, five different verdicts
# =============================================================================

DS = {
    "A": (
        np.array([0.10, 0.22, 0.40, 0.46, 0.52]),
        np.array([-1.0, -1.6, -6.0, -5.4, -1.0]),
    ),
    "B": (np.array([0.50]), np.array([1.0])),
}


def experiment_1():
    gp, grid = GP_MODEL, GRID
    Hx0, Hf0 = prior_entropies(gp, grid, RNG, n_samples=60000)
    rows = {}

    fig, axes = plt.subplots(
        2, 2, figsize=(COL2, 3.1), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12, "wspace": 0.14},
    )

    for j, name in enumerate(["A", "B"]):
        X, y = DS[name]
        mu, sd = gp.posterior_marginal(X, y, grid)
        mu_obs, _ = gp.posterior_marginal(X, y, X)

        # information gain is a Monte Carlo estimate: repeat it so the paper can
        # report a standard error alongside each figure.
        igx_s, igf_s = [], []
        for s in range(8):
            r = np.random.default_rng(100 + s)
            pc = prior_entropies(gp, grid, r, n_samples=60000)
            a, b = information_gain(gp, X, y, grid, r, n_samples=60000, prior_cache=pc)
            igx_s.append(a)
            igf_s.append(b)
        rows[name] = {
            "simple": simple_reward(gp, X, y, grid),
            "global": global_reward(gp, X, y, grid),
            "maxy": max_y(gp, X, y, grid),
            "cumulative": cumulative_reward(gp, X, y, grid),
            "ig_xstar": float(np.mean(igx_s)),
            "ig_xstar_se": float(np.std(igx_s) / np.sqrt(8)),
            "ig_fstar": float(np.mean(igf_s)),
            "ig_fstar_se": float(np.std(igf_s) / np.sqrt(8)),
            "n": int(np.size(X)),
        }

        x_sr = float(X[np.argmax(mu_obs)])
        x_gr = float(grid[np.argmax(mu)])

        ax = axes[0, j]
        ax.axhline(0.0, color=MUTED, lw=0.5, ls=(0, (4, 3)))
        band(ax, grid, mu, sd)
        ax.plot(X, y, "o", ms=4, mfc="white", mec=INK, mew=1.0, zorder=5,
                label="polls")
        ax.axvline(x_gr, color=C_GREEN, lw=2.0, ls="-", zorder=4,
                   label="global-reward rec.")
        ax.axvline(x_sr, color=C_ORANGE, lw=1.3, ls=(0, (3.5, 3.5)), zorder=6,
                   label="simple-reward rec.")
        ax.set_ylim(-8.6, 6.4)
        ax.set_title(
            f"Dataset {name}: "
            + ("five polls, all losses" if name == "A" else "one mild win"),
            loc="left", pad=3,
        )
        ax.text(
            0.985, 0.05,
            f"simple {rows[name]['simple']:+.2f}\nglobal {rows[name]['global']:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=INK,
            bbox=dict(fc="white", ec="#dddddd", lw=0.5, pad=2.2),
        )
        if j == 0:
            ax.set_ylabel("margin shift (points)")
            ax.legend(frameon=False, loc="upper left", fontsize=6.4,
                      handlelength=1.4, borderaxespad=0.3)

        # p(x* | D); boundary cells carry point masses that would flatten the
        # interior structure, so the axis is scaled to the interior.
        _, cov = gp.posterior(X, y, grid)
        paths = gp.sample_paths(mu, cov, 40000, RNG)
        idx = np.argmax(paths, axis=1)
        counts = np.bincount(idx, minlength=grid.size).astype(float)
        counts /= counts.sum()
        nb = 100  # coarser bins: the 501-cell histogram is Monte Carlo noise
        edges = np.linspace(0, 1, nb + 1)
        binned = np.add.reduceat(counts, np.searchsorted(grid, edges[:-1]))
        mass_lo, mass_hi = float(counts[0]), float(counts[-1])
        interior = binned[1:-1]
        axb = axes[1, j]
        axb.stairs(binned, edges, fill=True, color=C_PURPLE, alpha=0.9, lw=0)
        top = max(2e-3, float(interior.max()) * 1.3)
        axb.set_ylim(0, top)
        for edge, ha, m in ((0.005, "left", mass_lo), (0.995, "right", mass_hi)):
            axb.annotate(f"{m:.2f} at\nboundary", xy=(edge, top * 0.96),
                         fontsize=5.6, color=C_PURPLE, ha=ha, va="top",
                         linespacing=0.95)
        axb.set_xlabel("district coordinate $x$")
        if j == 0:
            axb.set_ylabel(r"$p(x^\ast\!\mid\!\mathcal{D})$")
        axb.tick_params(labelleft=False)

    fig.savefig("fig_datasets.pdf")
    plt.close(fig)

    RESULTS["e1"] = {"rows": rows, "H_prior_xstar": Hx0, "H_prior_fstar": Hf0}
    return rows


# =============================================================================
# E2 -- max y prefers the noisier dataset, and when it stops doing so
# =============================================================================


def experiment_2():
    grid = GRID
    rng = np.random.default_rng(7)

    gp_noisy = GP(lengthscale=0.08, amplitude=3.0, noise=3.0, mean=0.0)
    gp_clean = GP(lengthscale=0.08, amplitude=3.0, noise=0.15, mean=0.0)

    # Noisy scenario: the underlying campaign effect is flat (a genuinely
    # uninformative outcome); the apparent 6.2-point winner is polling noise.
    Xo = np.linspace(0.06, 0.94, 12)
    y_noisy = np.array(
        [0.4, -2.6, 1.9, -3.1, 2.2, -1.7, 6.2, -2.4, 1.1, -3.4, 2.8, -1.2]
    )
    # Clean scenario: a real 2.6-point effect, measured accurately.
    y_clean = 2.6 * np.exp(-0.5 * ((Xo - 0.62) / 0.10) ** 2) + 0.15 * rng.standard_normal(12)

    stats = {}
    for tag, gp, yy in [("noisy", gp_noisy, y_noisy), ("clean", gp_clean, y_clean)]:
        stats[tag] = {
            "maxy": max_y(gp, Xo, yy, grid),
            "simple": simple_reward(gp, Xo, yy, grid),
            "global": global_reward(gp, Xo, yy, grid),
            "noise": gp.sn,
            "snr": gp.sf / gp.sn,
        }

    # --- SNR sweep: how often does max-y rank two datasets the same way as
    #     simple reward?
    snrs = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 30])
    agree = []
    n_pairs = 400
    for snr in snrs:
        gp = GP(lengthscale=0.08, amplitude=3.0, noise=3.0 / snr, mean=0.0)
        hits = 0
        for _ in range(n_pairs):
            ds = []
            for _ in range(2):
                mu0, cov0 = gp.posterior(np.array([]), np.array([]), grid)
                f = gp.sample_paths(mu0, cov0, 1, rng)[0]
                loc = rng.choice(grid.size, size=8, replace=False)
                Xs, ys = grid[loc], f[loc] + gp.sn * rng.standard_normal(8)
                ds.append((Xs, ys))
            s = [simple_reward(gp, X, y, grid) for X, y in ds]
            m = [max_y(gp, X, y, grid) for X, y in ds]
            hits += int(np.sign(s[0] - s[1]) == np.sign(m[0] - m[1]))
        agree.append(hits / n_pairs)
    agree = np.array(agree)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.05),
                             gridspec_kw={"wspace": 0.42})
    for ax, tag, gp, yy in [
        (axes[0], "noisy", gp_noisy, y_noisy),
        (axes[1], "clean", gp_clean, y_clean),
    ]:
        mu, sd = gp.posterior_marginal(Xo, yy, grid)
        band(ax, grid, mu, sd)
        ax.plot(Xo, yy, "o", ms=3.5, mfc="white", mec=INK, mew=0.9, zorder=5)
        ax.axhline(stats[tag]["maxy"], color=C_ORANGE, lw=1.1, ls=(0, (3.5, 3.5)),
                   zorder=6)
        ax.axhline(stats[tag]["global"], color=C_GREEN, lw=1.1, zorder=4)
        ax.text(
            0.5, 0.02,
            r"$\max\mathbf{y}=$" + f"{stats[tag]['maxy']:.2f}\n"
            + f"global reward $=${stats[tag]['global']:.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2,
            linespacing=1.25,
            bbox=dict(fc="white", ec="#dddddd", lw=0.5, pad=2.0),
        )
        ax.set_ylim(-6.8, 7.6)
        ax.set_xlabel("district coordinate $x$")
        ax.set_title(
            rf"{tag}: $\sigma_n={gp.sn:g}$, SNR {gp.sf/gp.sn:.0f}", loc="left", pad=3
        )
    axes[0].set_ylabel("margin shift (points)")
    axes[1].tick_params(labelleft=False)
    axes[1].set_ylim(*axes[0].get_ylim())

    ax = axes[2]
    ax.plot(snrs, 100 * agree, "-o", color=C_BLUE, lw=1.4, ms=3.2)
    ax.axhline(95, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.text(30, 94.2, "95% agreement", fontsize=6.2, color=MUTED,
            va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "30"])
    ax.set_ylim(70, 103)
    ax.set_xlabel(r"signal-to-noise ratio $\sigma_f/\sigma_n$")
    ax.set_ylabel("agreement with simple\nreward (% of pairs)", fontsize=7)
    ax.set_title(r"when is $\max\mathbf{y}$ safe?", loc="left", pad=3)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)

    fig.savefig("fig_maxy.pdf")
    plt.close(fig)

    # smallest SNR at which agreement first reaches 95%
    thr = None
    for s, a in zip(snrs, agree):
        if a >= 0.95:
            thr = float(s)
            break

    RESULTS["e2"] = {
        "stats": stats,
        "snrs": snrs.tolist(),
        "agreement": agree.tolist(),
        "snr_95": thr,
        "n_pairs": n_pairs,
    }
    return stats


# =============================================================================
# E3 -- risk tolerance: what beta buys and what it costs
# =============================================================================


def experiment_3(n_rep=400, n_obs=6):
    gp, grid = GP_MODEL, GRID
    rng = np.random.default_rng(11)
    betas = np.linspace(-3.0, 3.0, 25)

    realized = np.zeros((n_rep, betas.size))
    post_mu = np.zeros((n_rep, betas.size))
    post_sd = np.zeros((n_rep, betas.size))

    mu0, cov0 = gp.posterior(np.array([]), np.array([]), grid)
    for r in range(n_rep):
        f = gp.sample_paths(mu0, cov0, 1, rng)[0]
        loc = rng.choice(grid.size, size=n_obs, replace=False)
        X, y = grid[loc], f[loc] + gp.sn * rng.standard_normal(n_obs)
        mu, sd = gp.posterior_marginal(X, y, grid)
        for b_i, b in enumerate(betas):
            i = int(np.argmax(mu + b * sd))
            realized[r, b_i] = f[i]
            post_mu[r, b_i] = mu[i]
            post_sd[r, b_i] = sd[i]

    mean_real = realized.mean(0)
    sd_real = realized.std(0)
    q10 = np.percentile(realized, 10, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.05), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    ax.plot(betas, mean_real, color=C_BLUE, lw=1.6, label="mean")
    ax.plot(betas, q10, color=C_ORANGE, lw=1.6, ls=(0, (3.5, 3)),
            label="10th percentile")
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    b_opt = betas[int(np.argmax(mean_real))]
    ax.plot([b_opt], [mean_real.max()], "o", ms=4, color=C_BLUE, zorder=5)
    ax.annotate(rf"$\beta^\ast\!=\!{b_opt:.2f}$", xy=(b_opt, mean_real.max()),
                xytext=(b_opt - 0.4, mean_real.max() + 0.45), fontsize=6.2,
                color=C_BLUE, ha="right")
    ax.set_ylim(-3.4, 4.6)
    ax.set_xlabel(r"risk tolerance $\beta$")
    ax.set_ylabel("true margin at the\nrecommended district", fontsize=7)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6, fontsize=6.4)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("realised outcome", loc="left", pad=3)

    ax = axes[1]
    ax.plot(betas, post_sd.mean(0), color=C_PURPLE, lw=1.6)
    ax.set_xlabel(r"risk tolerance $\beta$")
    ax.set_ylabel(r"posterior s.d. $\sigma_{\mathcal{D}}$" + "\nat the recommendation",
                  fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("what $\\beta$ actually controls", loc="left", pad=3)

    ax = axes[2]
    for b, col, lab in ((-2.0, C_ORANGE, r"$\beta=-2$ averse"),
                        (0.0, C_BLUE, r"$\beta=0$ neutral"),
                        (2.0, C_PURPLE, r"$\beta=+2$ seeking")):
        i = int(np.argmin(np.abs(betas - b)))
        v = np.sort(realized[:, i])
        ax.plot(v, np.linspace(0, 1, v.size), color=col, lw=1.5, label=lab)
    ax.set_xlim(-6, 8)
    ax.set_xlabel("realised margin (points)")
    ax.set_ylabel("empirical CDF", fontsize=7)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, fontsize=6.2)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("outcome distributions", loc="left", pad=3)

    fig.savefig("fig_risk.pdf")
    plt.close(fig)

    def at(b):
        i = int(np.argmin(np.abs(betas - b)))
        return {
            "mean": float(mean_real[i]),
            "sd": float(sd_real[i]),
            "q10": float(q10[i]),
            "post_sd": float(post_sd.mean(0)[i]),
        }

    RESULTS["e3"] = {
        "n_rep": n_rep, "n_obs": n_obs,
        "beta_-2": at(-2.0), "beta_0": at(0.0), "beta_+2": at(2.0),
        "argmax_mean_beta": float(betas[int(np.argmax(mean_real))]),
    }
    return RESULTS["e3"]


# =============================================================================
# E4 -- every policy wins under its own utility
# =============================================================================

POLICIES = {
    "EI": expected_improvement,
    "KG": knowledge_gradient,
    "GP-UCB": lambda gp, X, y, g: gp_ucb(gp, X, y, g, beta=2.0),
    "Random": None,
}
PCOL = {"EI": C_BLUE, "KG": C_ORANGE, "GP-UCB": C_GREEN, "Random": C_PURPLE}


def experiment_4(n_rep=30, n_iter=20, n_init=3, grid_n=201):
    gp = GP(lengthscale=0.08, amplitude=3.0, noise=0.5, mean=0.0)
    grid = np.linspace(0, 1, grid_n)
    rng = np.random.default_rng(2026)
    Hx0, Hf0 = prior_entropies(gp, grid, rng, n_samples=40000)

    traj = {p: {k: np.zeros((n_rep, n_iter + 1)) for k in ("simple", "global", "cum")}
            for p in POLICIES}
    ig_ckpt = list(range(0, n_iter + 1, 4))
    ig_traj = {p: np.zeros((n_rep, len(ig_ckpt))) for p in POLICIES}
    final = {p: {k: [] for k in ("simple", "global", "cum", "ig_x", "ig_f", "regret")}
             for p in POLICIES}

    mu0, cov0 = gp.posterior(np.array([]), np.array([]), grid)
    for r in range(n_rep):
        f = gp.sample_paths(mu0, cov0, 1, rng)[0]
        f_star = float(f.max())
        init = rng.choice(grid.size, size=n_init, replace=False)
        X0, y0 = grid[init], f[init] + gp.sn * rng.standard_normal(n_init)

        for pname, acq in POLICIES.items():
            X, y = X0.copy(), y0.copy()
            for t in range(n_iter + 1):
                traj[pname]["simple"][r, t] = simple_reward(gp, X, y, grid)
                traj[pname]["global"][r, t] = global_reward(gp, X, y, grid)
                traj[pname]["cum"][r, t] = cumulative_reward(gp, X, y, grid)
                if t in ig_ckpt:
                    ig_traj[pname][r, ig_ckpt.index(t)] = information_gain(
                        gp, X, y, grid, rng, n_samples=12000,
                        prior_cache=(Hx0, Hf0)
                    )[0]
                if t == n_iter:
                    break
                if acq is None:
                    i = int(rng.integers(grid.size))
                else:
                    a = acq(gp, X, y, grid)
                    i = int(np.argmax(a))
                X = np.append(X, grid[i])
                y = np.append(y, f[i] + gp.sn * rng.standard_normal())

            igx, igf = information_gain(
                gp, X, y, grid, rng, n_samples=20000, prior_cache=(Hx0, Hf0)
            )
            mu, _ = gp.posterior_marginal(X, y, grid)
            x_rec = int(np.argmax(mu))
            final[pname]["simple"].append(simple_reward(gp, X, y, grid))
            final[pname]["global"].append(global_reward(gp, X, y, grid))
            final[pname]["cum"].append(cumulative_reward(gp, X, y, grid))
            final[pname]["ig_x"].append(igx)
            final[pname]["ig_f"].append(igf)
            final[pname]["regret"].append(f_star - float(f[x_rec]))
        print(f"  E4 replicate {r+1}/{n_rep}", flush=True)

    labels = {"simple": "simple reward", "cum": "cumulative reward"}
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.05), gridspec_kw={"wspace": 0.36})
    it = np.arange(n_iter + 1)
    for ax, key in zip(axes[:2], ["simple", "cum"]):
        for pname in POLICIES:
            m = traj[pname][key].mean(0)
            se = traj[pname][key].std(0) / np.sqrt(n_rep)
            ax.plot(it, m, color=PCOL[pname], lw=1.5, label=pname)
            ax.fill_between(it, m - se, m + se, color=PCOL[pname], alpha=0.16, lw=0)
        ax.set_xlabel("observations")
        ax.set_title(labels[key], loc="left", pad=3)
        ax.grid(True, alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("utility (margin points)", fontsize=7)
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.4,
                   ncol=2, fontsize=6.2, columnspacing=1.0)

    ax = axes[2]
    for pname in POLICIES:
        m = ig_traj[pname].mean(0)
        se = ig_traj[pname].std(0) / np.sqrt(n_rep)
        ax.plot(ig_ckpt, m, "-o", ms=2.8, color=PCOL[pname], lw=1.5)
        ax.fill_between(ig_ckpt, m - se, m + se, color=PCOL[pname], alpha=0.16, lw=0)
    ax.set_xlabel("observations")
    ax.set_ylabel("nats", fontsize=7)
    ax.set_title(r"info. gain about $x^\ast$", loc="left", pad=3)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)

    ax = axes[3]
    names = list(POLICIES)
    for i, pname in enumerate(names):
        v = np.array(final[pname]["regret"])
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(n_rep)
        ax.errorbar([i], [m], yerr=[1.96 * se], fmt="o", ms=4.5,
                    color=PCOL[pname], capsize=2.5, lw=1.3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("regret (points)", fontsize=7)
    ax.set_title("terminal regret", loc="left", pad=3)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(names) - 0.4)

    fig.savefig("fig_policy.pdf")
    plt.close(fig)
    np.savez("e4_traj.npz", ig_ckpt=np.array(ig_ckpt),
             **{f"{p}_{k}": traj[p][k] for p in POLICIES for k in traj[p]},
             **{f"{p}_igtraj": ig_traj[p] for p in POLICIES},
             **{f"{p}_regret": np.array(final[p]["regret"]) for p in POLICIES})

    summ = {}
    for pname in POLICIES:
        summ[pname] = {
            k: [float(np.mean(v)), float(np.std(v) / np.sqrt(n_rep))]
            for k, v in final[pname].items()
        }
    winners = {
        k: max(summ, key=lambda p: summ[p][k][0])
        for k in ("simple", "global", "cum", "ig_x", "ig_f")
    }
    winners["regret"] = min(summ, key=lambda p: summ[p]["regret"][0])

    # All policies see the same objective within a replicate, so paired
    # differences against Random are far more powerful than unpaired means.
    paired = {}
    for k in ("simple", "global", "cum", "ig_x", "ig_f", "regret"):
        base = np.array(final["Random"][k])
        paired[k] = {}
        for pname in POLICIES:
            d = np.array(final[pname][k]) - base
            paired[k][pname] = [
                float(d.mean()), float(d.std(ddof=1) / np.sqrt(n_rep))
            ]
    # Also pair the best-vs-second-best contrast within each metric.
    contrasts = {}
    for k in ("simple", "global", "cum", "ig_x", "ig_f", "regret"):
        sign = -1.0 if k == "regret" else 1.0
        order = sorted(POLICIES, key=lambda p: sign * summ[p][k][0], reverse=True)
        a, b = order[0], order[1]
        d = sign * (np.array(final[a][k]) - np.array(final[b][k]))
        contrasts[k] = {
            "best": a, "second": b,
            "diff": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(n_rep)),
            "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(n_rep))),
        }

    RESULTS["e4"] = {"summary": summ, "winners": winners, "paired_vs_random": paired,
                     "contrasts": contrasts, "raw": {p: {k: list(map(float, v))
                     for k, v in final[p].items()} for p in POLICIES},
                     "n_rep": n_rep, "n_iter": n_iter, "n_init": n_init}
    return summ, winners


if __name__ == "__main__":
    print("E1 ...", flush=True)
    print(json.dumps(experiment_1(), indent=1, default=float))
    print("E2 ...", flush=True)
    print(json.dumps(experiment_2(), indent=1, default=float))
    print("E3 ...", flush=True)
    print(json.dumps(experiment_3(), indent=1, default=float))
    print("E4 ...", flush=True)
    s, w = experiment_4()
    print(json.dumps(s, indent=1, default=float))
    print("winners:", w)
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print("wrote results.json")
