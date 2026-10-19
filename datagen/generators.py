"""
datagen/generators.py
=====================
Deterministic generators for the five Phase 1 directed graph families.

All randomness comes from one numpy Generator seeded with the graph's explicit
seed, so the same (family, tier, seed, parameters) always yields byte-identical
edges regardless of machine or Spark partitioning. Generation is pure numpy;
Spark is only used afterwards to validate that the Parquet output loads.

Edge semantics (identical for every family, see config.EDGE_POLICY):
  directed src -> dst, int32 ids in [0, n), no self-loops, no duplicate pairs.
"""

from typing import Callable, Dict, Tuple

import numpy as np

from datagen import config

Edges = Tuple[np.ndarray, np.ndarray]


def _encode(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return (src.astype(np.int64) << 32) | dst.astype(np.int64)


def _decode(keys: np.ndarray) -> Edges:
    return (keys >> 32).astype(np.int32), (keys & 0xFFFFFFFF).astype(np.int32)


def clean_edges(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Drop self-loops and duplicate pairs. Returns sorted unique keys and drop counts."""
    raw = int(len(src))
    keep = src != dst
    self_loops = raw - int(keep.sum())
    keys = _encode(src[keep], dst[keep])
    uniq = np.unique(keys)
    return uniq, {
        "raw_edges_sampled": raw,
        "self_loops_removed": self_loops,
        "duplicates_removed": int(len(keys) - len(uniq)),
    }


def sample_to_target(
    rng: np.random.Generator,
    m_target: int,
    sampler: Callable[[int], Edges],
    oversample: float = 1.03,
    max_rounds: int = 20,
) -> Tuple[Edges, Dict[str, int]]:
    """
    Draw edges from `sampler` until at least m_target unique non-loop edges exist,
    then keep a seeded random subset of exactly m_target. Reports cleaning counts.
    """
    pool = np.empty(0, dtype=np.int64)
    totals = {"raw_edges_sampled": 0, "self_loops_removed": 0, "duplicates_removed": 0}
    need = int(m_target * oversample)
    for round_idx in range(max_rounds):
        s, d = sampler(need)
        keys, info = clean_edges(s, d)
        totals["raw_edges_sampled"] += info["raw_edges_sampled"]
        totals["self_loops_removed"] += info["self_loops_removed"]
        merged = np.unique(np.concatenate([pool, keys]))
        totals["duplicates_removed"] += info["duplicates_removed"] + (len(pool) + len(keys) - len(merged))
        pool = merged
        if len(pool) >= m_target:
            break
        need = int((m_target - len(pool)) * 1.5) + 1000
    else:
        raise RuntimeError(
            f"Could not reach {m_target} unique edges after {max_rounds} rounds "
            f"(have {len(pool)}); the family parameters are too concentrated for this size."
        )
    totals["sampling_rounds"] = round_idx + 1
    totals["unique_edges_before_subsample"] = int(len(pool))
    if len(pool) > m_target:
        pick = np.sort(rng.choice(len(pool), size=m_target, replace=False))
        pool = pool[pick]
    return _decode(pool), totals


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def gen_uniform(rng, n, avg_degree, params, tier) -> Tuple[Edges, Dict, Dict]:
    """Every vertex: out-degree k (k copies of each id) and in-degree k (k random permutations)."""
    k = int(avg_degree)
    src = np.tile(np.arange(n, dtype=np.int32), k)
    dst = np.concatenate([rng.permutation(n).astype(np.int32) for _ in range(k)])
    keys, info = clean_edges(src, dst)
    info["sampling_rounds"] = 1
    info["unique_edges_before_subsample"] = int(len(keys))
    meta = {"construction": f"{k} rounds of (i -> perm[i]) over all vertices"}
    return _decode(keys), info, meta


def _powerlaw_probs(n: int, gamma: float, max_share: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = ranks ** (-1.0 / (gamma - 1.0))
    p = w / w.sum()
    # Cap each vertex's share, then renormalise; repeat so the cap still holds.
    for _ in range(50):
        over = p > max_share
        if not over.any():
            break
        p = np.minimum(p, max_share)
        p /= p.sum()
    return p


def gen_powerlaw_balanced(rng, n, avg_degree, params, tier):
    m_target = n * avg_degree
    p_rank = _powerlaw_probs(n, params["gamma"], params["max_share"])
    perm = rng.permutation(n)            # random ids so hubs are not ids 0..k
    p = np.empty(n)
    p[perm] = p_rank                     # same weight for in and out -> balanced hubs
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    def sampler(size):
        s = np.searchsorted(cdf, rng.random(size), side="right").astype(np.int32)
        d = np.searchsorted(cdf, rng.random(size), side="right").astype(np.int32)
        return s, d

    edges, info = sample_to_target(rng, m_target, sampler)
    meta = {"top_weight_vertices": perm[:10].tolist(), "expected_max_share": float(p_rank.max())}
    return edges, info, meta


def _hub_side_heavy(rng, n, avg_degree, params, heavy_side: str):
    m_target = n * avg_degree
    hub_count = max(1, int(round(params["hubs_per_100k"] * n / 100_000)))
    hubs = rng.choice(n, size=hub_count, replace=False).astype(np.int32)
    z = 1.0 / np.arange(1, hub_count + 1, dtype=np.float64) ** params["hub_zipf_s"]
    hub_cdf = np.cumsum(z / z.sum())
    hub_cdf[-1] = 1.0
    frac = params["hub_edge_fraction"]

    def sampler(size):
        a = rng.integers(0, n, size=size, dtype=np.int32)       # uniform side
        b = rng.integers(0, n, size=size, dtype=np.int32)       # uniform unless hub edge
        is_hub = rng.random(size) < frac
        b[is_hub] = hubs[np.searchsorted(hub_cdf, rng.random(int(is_hub.sum())), side="right")]
        return (a, b) if heavy_side == "in" else (b, a)

    edges, info = sample_to_target(rng, m_target, sampler)
    meta = {"hub_count": hub_count, "hub_ids": hubs.tolist()}
    return edges, info, meta


def gen_in_heavy(rng, n, avg_degree, params, tier):
    return _hub_side_heavy(rng, n, avg_degree, params, "in")


def gen_out_heavy(rng, n, avg_degree, params, tier):
    return _hub_side_heavy(rng, n, avg_degree, params, "out")


def gen_multi_hub(rng, n, avg_degree, params, tier):
    m_target = n * avg_degree
    hub_count = int(params["hub_count"][tier])
    base = params["hub_degree"]
    jit = params["hub_degree_jitter"]
    hubs = rng.choice(n, size=hub_count, replace=False).astype(np.int32)
    d_in = np.round(base * rng.uniform(1 - jit, 1 + jit, hub_count)).astype(np.int64)
    d_out = np.round(base * rng.uniform(1 - jit, 1 + jit, hub_count)).astype(np.int64)

    # Hub edges: distinct partners per hub (choice without replacement), excluding the hub itself.
    hub_src, hub_dst = [], []
    for h, di, do in zip(hubs, d_in, d_out):
        ins = rng.choice(n - 1, size=int(di), replace=False)
        ins = np.where(ins >= h, ins + 1, ins).astype(np.int32)
        outs = rng.choice(n - 1, size=int(do), replace=False)
        outs = np.where(outs >= h, outs + 1, outs).astype(np.int32)
        hub_src += [ins, np.full(len(outs), h, np.int32)]
        hub_dst += [np.full(len(ins), h, np.int32), outs]
    hub_keys, _ = clean_edges(np.concatenate(hub_src), np.concatenate(hub_dst))
    m_bg = m_target - len(hub_keys)

    def sampler(size):
        return (rng.integers(0, n, size=size, dtype=np.int32),
                rng.integers(0, n, size=size, dtype=np.int32))

    (bs, bd), info = sample_to_target(rng, m_bg, sampler)
    keys = np.unique(np.concatenate([hub_keys, _encode(bs, bd)]))
    info["duplicates_removed"] += int(len(hub_keys) + len(bs) - len(keys))
    info["background_edges"] = int(len(bs))
    info["hub_edges"] = int(len(hub_keys))
    meta = {
        "hub_count": hub_count,
        "hub_ids": hubs.tolist(),
        "planned_hub_in_degree": d_in.tolist(),
        "planned_hub_out_degree": d_out.tolist(),
    }
    return _decode(keys), info, meta


GENERATORS = {
    "uniform": gen_uniform,
    "powerlaw_balanced": gen_powerlaw_balanced,
    "in_heavy": gen_in_heavy,
    "out_heavy": gen_out_heavy,
    "multi_hub": gen_multi_hub,
}


def resolve_params(family: str, tier: str) -> Dict:
    """Family parameters with any per-tier values resolved for this tier."""
    out = {}
    for k, v in config.FAMILY_PARAMS[family].items():
        out[k] = v[tier] if isinstance(v, dict) else v
    return out


def generate(family: str, tier: str, seed: int) -> Tuple[Edges, Dict]:
    """Generate one graph. Returns ((src, dst), generation_info)."""
    if family not in GENERATORS:
        raise ValueError(f"Unknown family {family!r}; expected one of {config.FAMILIES}")
    t = config.TIERS[tier]
    n, k = t["num_vertices"], t["avg_degree"]
    rng = np.random.default_rng(seed)
    (src, dst), cleaning, meta = GENERATORS[family](rng, n, k, config.FAMILY_PARAMS[family], tier)
    # Canonical order (sorted by src, dst) so the Parquet bytes are reproducible.
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    return (src, dst), {"cleaning": cleaning, "family_meta": meta}
