"""Degree-preserving null model: one Curveball Markov chain per cuisine (numba).

Algorithm (Strona et al., 2014, *Nature Communications* 5:4114; Carstens, 2015)
---------------------------------------------------------------------------
The recipe x ingredient incidence matrix B (binary) is stored as one ingredient
list per recipe. One *trade* picks two distinct recipes i, j uniformly at random,
computes the ingredients unique to each (A = i\\j, B = j\\i), shuffles the pool
A ∪ B and gives |A| of them back to i and the rest to j. Shared ingredients stay.
Every trade preserves each recipe's ingredient count (row sums), each
ingredient's recipe frequency (column sums) and binarity (no duplicates).

Chain layout (per cuisine): burn-in of ``burn_in_multiplier * n_recipes`` trades,
then ``n_randomizations`` samples with ``thinning_multiplier * n_recipes`` trades
between consecutive samples. For each sample the co-occurrence counts among
eligible ingredients (equivalent of B_e^T B_e) are accumulated as exact integer
sums and sums of squares, so no randomised matrix is stored.

The chain runs on the *full* incidence matrix of the cuisine (all normalised
ingredients, not only eligible ones) so that each recipe's complete ingredient
count is preserved; only eligible-pair co-occurrences are accumulated.

Mixing is assessed (not assumed) by (i) the burn-in trace of a summary statistic
S = Σ_{a<b} n(a,b)^2 over eligible pairs, (ii) the autocorrelation function and
effective sample size of S and of the co-count of the most frequent pair across
the retained samples, and (iii) a Geweke-style comparison of early vs late samples.

Usage::

    python src/null_model.py --check       # degree preservation / reproducibility checks
    python src/null_model.py --benchmark   # mandatory timing benchmark on real data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
from numba import njit

from src.common import get_logger, load_config, rpath, sha256_obj, write_json

log = get_logger()


# =========================================================================== numba kernels
@njit(cache=True)
def _trade(indptr, indices, i, j, mark, pool, common):
    """One Curveball trade between rows i and j. Returns 0 trivial, 1 nontrivial-unchanged, 2 changed."""
    si, ei = indptr[i], indptr[i + 1]
    sj, ej = indptr[j], indptr[j + 1]
    for k in range(si, ei):
        mark[indices[k]] = 1
    nb = 0
    nc = 0
    # j-only items appended after i-only items; collect j-only first in a temp region at the end
    for k in range(sj, ej):
        x = indices[k]
        if mark[x] == 1:
            mark[x] = 2
            common[nc] = x
            nc += 1
    na = 0
    for k in range(si, ei):
        x = indices[k]
        if mark[x] == 1:
            pool[na] = x
            na += 1
    for k in range(sj, ej):
        x = indices[k]
        if mark[x] == 0:
            pool[na + nb] = x
            mark[x] = 3
            nb += 1
    status = 0
    if na > 0 and nb > 0:
        total = na + nb
        for k in range(total - 1, 0, -1):  # Fisher-Yates shuffle
            r = np.random.randint(0, k + 1)
            tmp = pool[k]
            pool[k] = pool[r]
            pool[r] = tmp
        changed = False
        for k in range(na):
            if mark[pool[k]] == 3:
                changed = True
                break
        status = 2 if changed else 1
        p = si
        for k in range(nc):
            indices[p] = common[k]
            p += 1
        for k in range(na):
            indices[p] = pool[k]
            p += 1
        p = sj
        for k in range(nc):
            indices[p] = common[k]
            p += 1
        for k in range(na, total):
            indices[p] = pool[k]
            p += 1
    # reset marks (all items of both rows)
    for k in range(si, ei):
        mark[indices[k]] = 0
    for k in range(sj, ej):
        mark[indices[k]] = 0
    return status


@njit(cache=True)
def _cooc(indptr, indices, elig, cnt, buf):
    """Fill upper-triangular eligible co-occurrence counts into ``cnt`` (zeroed here)."""
    ne = cnt.shape[0]
    for a in range(ne):
        for b in range(ne):
            cnt[a, b] = 0
    nrows = indptr.shape[0] - 1
    for r in range(nrows):
        m = 0
        for k in range(indptr[r], indptr[r + 1]):
            e = elig[indices[k]]
            if e >= 0:
                buf[m] = e
                m += 1
        for p in range(m):
            for q in range(p + 1, m):
                a = buf[p]
                b = buf[q]
                if a < b:
                    cnt[a, b] += 1
                else:
                    cnt[b, a] += 1


@njit(cache=True)
def _sumsq_upper(cnt):
    ne = cnt.shape[0]
    s = 0.0
    for a in range(ne):
        for b in range(a + 1, ne):
            s += float(cnt[a, b]) * float(cnt[a, b])
    return s


@njit(cache=True)
def _run_chain(indptr, indices0, elig, ne, n_cols, n_samples, burn_in, thin, seed, record_every, pa, pb,
               do_accumulate):
    np.random.seed(seed)
    indices = indices0.copy()
    nrows = indptr.shape[0] - 1
    maxdeg = 0
    for r in range(nrows):
        d = indptr[r + 1] - indptr[r]
        if d > maxdeg:
            maxdeg = d
    mark = np.zeros(n_cols, dtype=np.int8)
    pool = np.empty(2 * maxdeg + 1, dtype=indices.dtype)
    common = np.empty(maxdeg + 1, dtype=indices.dtype)
    buf = np.empty(maxdeg + 1, dtype=np.int64)
    cnt = np.zeros((max(ne, 1), max(ne, 1)), dtype=np.int64)
    s1 = np.zeros((max(ne, 1), max(ne, 1)), dtype=np.int64)
    s2 = np.zeros((max(ne, 1), max(ne, 1)), dtype=np.int64)
    n_rec = burn_in // record_every if record_every > 0 else 0
    burn_trace = np.zeros(n_rec, dtype=np.float64)
    sample_S = np.zeros(n_samples, dtype=np.float64)
    sample_pair = np.zeros(n_samples, dtype=np.float64)
    stats = np.zeros(3, dtype=np.int64)  # attempted, nontrivial, changed
    if nrows < 2:
        return s1, s2, burn_trace, sample_S, sample_pair, stats, indices
    rec = 0
    for t in range(burn_in):
        i = np.random.randint(0, nrows)
        j = np.random.randint(0, nrows - 1)
        if j >= i:
            j += 1
        st = _trade(indptr, indices, i, j, mark, pool, common)
        stats[0] += 1
        if st > 0:
            stats[1] += 1
        if st == 2:
            stats[2] += 1
        if record_every > 0 and (t + 1) % record_every == 0 and rec < n_rec:
            _cooc(indptr, indices, elig, cnt, buf)
            burn_trace[rec] = _sumsq_upper(cnt)
            rec += 1
    for s in range(n_samples):
        for t in range(thin):
            i = np.random.randint(0, nrows)
            j = np.random.randint(0, nrows - 1)
            if j >= i:
                j += 1
            st = _trade(indptr, indices, i, j, mark, pool, common)
            stats[0] += 1
            if st > 0:
                stats[1] += 1
            if st == 2:
                stats[2] += 1
        if do_accumulate:
            _cooc(indptr, indices, elig, cnt, buf)
            for a in range(ne):
                for b in range(a + 1, ne):
                    c = cnt[a, b]
                    s1[a, b] += c
                    s2[a, b] += c * c
            sample_S[s] = _sumsq_upper(cnt)
            if pa >= 0:
                sample_pair[s] = cnt[pa, pb]
    return s1, s2, burn_trace, sample_S, sample_pair, stats, indices


# =========================================================================== python API
def curveball_randomize(B: sp.csr_matrix, n_trades: int, seed: int) -> sp.csr_matrix:
    """Return one randomised copy of binary CSR matrix ``B`` after ``n_trades`` Curveball trades."""
    B = sp.csr_matrix(B)
    B.sort_indices()
    indptr = B.indptr.astype(np.int64)
    indices = B.indices.astype(np.int64)
    elig = -np.ones(B.shape[1], dtype=np.int64)
    _, _, _, _, _, _, out = _run_chain(indptr, indices, elig, 0, B.shape[1], 0, int(n_trades), 0, int(seed), 0, -1, -1,
                                       False)
    R = sp.csr_matrix((np.ones(len(out), dtype=np.int32), out, indptr), shape=B.shape)
    R.sort_indices()
    return R


def chain_seed(base_seed: int, cuisine_index: int, scope: str) -> int:
    """Deterministic per-chain seed derived from the global seed (scope: 'descriptive'|'predictive')."""
    offset = {"descriptive": 0, "predictive": 500_000, "benchmark": 900_000}[scope]
    return (int(base_seed) * 1000 + offset + 1000 * (cuisine_index + 1)) % 1_000_000


def acf(x: np.ndarray, max_lag: int) -> list[float]:
    """Sample autocorrelation function for lags 1..max_lag."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3 or np.var(x) == 0:
        return [float("nan")] * max_lag
    x = x - x.mean()
    denom = np.dot(x, x)
    return [float(np.dot(x[:-k], x[k:]) / denom) if k < n else float("nan") for k in range(1, max_lag + 1)]


def effective_sample_size(x: np.ndarray, max_lag: int = 200) -> float:
    """ESS = n / (1 + 2 Σ ρ_k), summing ρ_k until the first non-positive value (Geyer initial positive)."""
    n = len(x)
    rho = acf(x, min(max_lag, n - 1))
    s = 0.0
    for r in rho:
        if not np.isfinite(r) or r <= 0:
            break
        s += r
    return float(n / (1 + 2 * s))


def geweke_z(x: np.ndarray) -> float:
    """Geweke-style z comparing the first 10% and last 50% of samples (naive SE)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 20:
        return float("nan")
    a, b = x[: n // 10], x[n // 2:]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def run_null(B_full: sp.csr_matrix, eligible_cols: np.ndarray, n_samples: int, burn_in: int, thin: int, seed: int,
             max_lag: int = 20, record_points: int = 20) -> dict:
    """Run one Curveball chain and return null mean/std matrices plus diagnostics.

    ``eligible_cols`` are the column indices (into ``B_full``) of eligible ingredients,
    in the order defining the returned (ne x ne, upper-triangular) matrices.
    """
    B = sp.csr_matrix(B_full)
    B.sort_indices()
    if B.data.size and (B.data.max() > 1):
        raise ValueError("Incidence matrix must be binary")
    indptr = B.indptr.astype(np.int64)
    indices = B.indices.astype(np.int64)
    ne = len(eligible_cols)
    elig = -np.ones(B.shape[1], dtype=np.int64)
    elig[np.asarray(eligible_cols, dtype=np.int64)] = np.arange(ne)
    # observed co-occurrence among eligible columns
    Be = B[:, eligible_cols]
    obs = (Be.T @ Be).toarray().astype(np.int64)
    obs_u = np.triu(obs, 1)
    obs_S = float((obs_u.astype(float) ** 2).sum())
    if ne >= 2:
        pa, pb = np.unravel_index(int(np.argmax(obs_u)), obs_u.shape)
    else:
        pa, pb = -1, -1
    record_every = max(1, burn_in // record_points) if burn_in > 0 else 0
    t0 = time.perf_counter()
    s1, s2, burn_trace, sample_S, sample_pair, stats, _ = _run_chain(
        indptr, indices, elig, ne, B.shape[1], int(n_samples), int(burn_in), int(thin), int(seed), int(record_every),
        int(pa), int(pb), True)
    elapsed = time.perf_counter() - t0
    n = int(n_samples)
    s1 = s1[:ne, :ne].astype(np.int64)
    s2 = s2[:ne, :ne].astype(np.int64)
    mean = s1 / max(n, 1)
    if n >= 2:
        num = n * s2 - s1 * s1  # exact integer arithmetic
        var = num / (n * (n - 1))
        var[var < 0] = 0.0
        std = np.sqrt(var)
    else:
        std = np.zeros_like(mean)
    diag = {
        "algorithm": "curveball",
        "n_recipes": int(B.shape[0]), "n_ingredients_full": int(B.shape[1]), "n_eligible": int(ne),
        "n_incidences": int(B.nnz),
        "n_samples": n, "burn_in_trades": int(burn_in), "thinning_trades": int(thin), "seed": int(seed),
        "attempted_trades": int(stats[0]), "nontrivial_trades": int(stats[1]), "changed_trades": int(stats[2]),
        "chain_seconds": round(elapsed, 3),
        "observed_S": obs_S,
        "burn_in_trace_S": burn_trace.tolist(), "burn_in_record_every": int(record_every),
        "sample_S_mean": float(sample_S.mean()) if n else float("nan"),
        "sample_S_sd": float(sample_S.std(ddof=1)) if n > 1 else float("nan"),
        "acf_S": acf(sample_S, max_lag) if n > 2 else [],
        "ess_S": effective_sample_size(sample_S) if n > 2 else float("nan"),
        "geweke_z_S": geweke_z(sample_S),
        "tracked_pair_index": [int(pa), int(pb)],
        "acf_pair": acf(sample_pair, max_lag) if n > 2 else [],
        "ess_pair": effective_sample_size(sample_pair) if n > 2 else float("nan"),
        "geweke_z_pair": geweke_z(sample_pair),
    }
    if burn_trace.size >= 4 and n > 1:
        tail = burn_trace[len(burn_trace) // 2:]
        diag["burn_in_second_half_mean_minus_sample_mean_in_sd"] = float(
            (tail.mean() - sample_S.mean()) / sample_S.std(ddof=1)) if sample_S.std(ddof=1) > 0 else float("nan")
    return {"observed": obs_u, "mean": mean, "std": std, "s1": s1, "s2": s2, "sample_S": sample_S,
            "sample_pair": sample_pair, "diagnostics": diag}


# =========================================================================== caching
def null_cache_key(payload: dict) -> str:
    src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return sha256_obj({**payload, "null_model_source": src})


def run_null_cached(B_full, eligible_cols, n_samples, burn_in, thin, seed, cache_dir: Path | None, key_payload: dict,
                    max_lag: int = 20) -> tuple[dict, bool]:
    """Run :func:`run_null`, re-using a cached result only if the full hash key matches."""
    key = null_cache_key({**key_payload, "n_samples": n_samples, "burn_in": burn_in, "thin": thin, "seed": seed,
                          "eligible_cols": sha256_obj(list(map(int, eligible_cols))),
                          "B_hash": hashlib.sha256(np.ascontiguousarray(B_full.indices).tobytes()
                                                   + np.ascontiguousarray(B_full.indptr).tobytes()).hexdigest()})
    if cache_dir is not None:
        f = Path(cache_dir) / f"null_{key[:24]}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            diag = json.loads(str(z["diag"]))
            diag["cache_hit"] = True
            return {"observed": z["observed"], "mean": z["mean"], "std": z["std"], "s1": z["s1"], "s2": z["s2"],
                    "sample_S": z["sample_S"], "sample_pair": z["sample_pair"], "diagnostics": diag}, True
    res = run_null(B_full, eligible_cols, n_samples, burn_in, thin, seed, max_lag=max_lag)
    res["diagnostics"]["cache_hit"] = False
    res["diagnostics"]["cache_key"] = key
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache_dir) / f"null_{key[:24]}.npz", observed=res["observed"], mean=res["mean"],
                            std=res["std"], s1=res["s1"], s2=res["s2"], sample_S=res["sample_S"],
                            sample_pair=res["sample_pair"], diag=json.dumps(res["diagnostics"]))
    return res, False


# =========================================================================== checks & benchmark
def check_randomization(B: sp.csr_matrix, n_trades: int, seed: int) -> dict:
    """Verify degree preservation, binarity and seed reproducibility on matrix ``B``."""
    B = sp.csr_matrix(B)
    B.sort_indices()
    R1 = curveball_randomize(B, n_trades, seed)
    R2 = curveball_randomize(B, n_trades, seed)
    R3 = curveball_randomize(B, n_trades, seed + 1)
    rows_ok = bool(np.array_equal(np.diff(R1.indptr), np.diff(B.indptr)))
    cols_ok = bool(np.array_equal(np.bincount(R1.indices, minlength=B.shape[1]),
                                  np.bincount(B.indices, minlength=B.shape[1])))
    binary_ok = bool(all(len(set(R1.indices[R1.indptr[r]:R1.indptr[r + 1]])) == R1.indptr[r + 1] - R1.indptr[r]
                         for r in range(B.shape[0])))
    repro = bool((R1 != R2).nnz == 0)
    moved = int((R1 != B).nnz // 2)
    return {"row_degrees_preserved": rows_ok, "col_degrees_preserved": cols_ok, "binary": binary_ok,
            "reproducible_same_seed": repro, "different_seed_differs": bool((R1 != R3).nnz > 0),
            "cells_moved": moved, "n_trades": int(n_trades), "shape": list(B.shape),
            "passed": rows_ok and cols_ok and binary_ok and repro}


def warmup() -> float:
    """Trigger numba compilation on a tiny matrix; returns seconds spent."""
    t0 = time.perf_counter()
    B = sp.csr_matrix(np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=np.int32))
    run_null(B, np.array([0, 1, 2]), 3, 10, 2, 1)
    curveball_randomize(B, 5, 1)
    return time.perf_counter() - t0


def benchmark(nets_inputs: dict, cfg: dict) -> dict:
    """Mandatory pre-run benchmark on the real incidence matrices.

    ``nets_inputs[c] = (B_full, eligible_cols, n_observed_pairs)``. Measures the burn-in time
    and the time for ``benchmark_samples`` samples, then *estimates* the full-run time.
    """
    nm = cfg["null_model"]
    nb = int(nm["benchmark_samples"])
    n_full = int(nm["n_randomizations"])
    compile_s = warmup()
    out = {"numba_compile_seconds": round(compile_s, 3), "benchmark_samples": nb, "target_samples": n_full,
           "per_cuisine": {}, "estimate_label": "ESTIMATE (extrapolated, not a measured full-run time)"}
    total_est = 0.0
    for idx, (c, (B, elig, n_obs)) in enumerate(nets_inputs.items()):
        nrec = B.shape[0]
        burn = int(nm["burn_in_multiplier"]) * nrec
        thin = int(nm["thinning_multiplier"]) * nrec
        seed = chain_seed(cfg["seed"], idx, "benchmark")
        t0 = time.perf_counter()
        r0 = run_null(B, elig, 0, burn, thin, seed)
        t_burn = time.perf_counter() - t0
        t0 = time.perf_counter()
        r1 = run_null(B, elig, nb, burn, thin, seed)
        t_all = time.perf_counter() - t0
        per_sample = max(t_all - t_burn, 0.0) / nb
        est = t_burn + per_sample * n_full
        total_est += est
        d = r1["diagnostics"]
        out["per_cuisine"][c] = {
            "matrix_shape": [int(B.shape[0]), int(B.shape[1])], "incidences": int(B.nnz),
            "eligible_ingredients": int(len(elig)), "eligible_pairs": int(len(elig) * (len(elig) - 1) // 2),
            "observed_pairs": int(n_obs), "burn_in_trades": burn, "thinning_trades": thin,
            "measured_burn_in_seconds": round(t_burn, 3), "measured_seconds_with_samples": round(t_all, 3),
            "measured_per_sample_seconds": round(per_sample, 5),
            "attempted_trades": d["attempted_trades"], "nontrivial_trades": d["nontrivial_trades"],
            "changed_trades": d["changed_trades"],
            "estimated_full_run_seconds": round(est, 2),
            "extrapolation": f"t_burn + n_samples * (t_with_{nb}_samples - t_burn)/{nb}",
        }
        del r0
    out["estimated_total_seconds_all_cuisines"] = round(total_est, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Curveball null model utilities: correctness checks and benchmark "
                                             "on the real descriptive incidence matrices.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--check", action="store_true", help="Run degree-preservation and reproducibility checks")
    ap.add_argument("--benchmark", action="store_true", help="Run the timing benchmark and save JSON")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"seed": args.seed})
    if not (args.check or args.benchmark):
        ap.print_help()
        return 0
    from src.backbone import incidence_inputs
    from src.preprocess import load_processed
    recipes, _ = load_processed(cfg)
    inputs = incidence_inputs(recipes, cfg)
    d2 = rpath(cfg["outputs"]["d2"])
    d2.mkdir(parents=True, exist_ok=True)
    if args.check:
        res = {c: check_randomization(B, 5 * B.shape[0], cfg["seed"]) for c, (B, _, _) in inputs.items()}
        write_json(res, d2 / "null_model_checks.json")
        log.info("Checks: %s", {c: r["passed"] for c, r in res.items()})
    if args.benchmark:
        res = benchmark(inputs, cfg)
        write_json(res, d2 / "null_benchmark.json")
        log.info("Benchmark estimate (all cuisines): %.1f s", res["estimated_total_seconds_all_cuisines"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
