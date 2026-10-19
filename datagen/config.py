"""
datagen/config.py
=================
Single source of truth for Phase 1 synthetic graph generation.

Every value here feeds the content hash in the graph_id, so changing a parameter
produces a new graph_id instead of silently altering an existing graph.
"""

GENERATOR_VERSION = "phase1-gen-v1"

# Seeds for the full 5 families x 3 tiers x 4 seeds = 60-instance matrix.
SEEDS = [101, 202, 303, 404]

# Size tiers. avg_degree is the target mean out-degree (= mean in-degree), so
# m_target = num_vertices * avg_degree, which keeps every tier inside the
# 5-10 edges-per-vertex band from the Phase 1 brief.
TIERS = {
    "small": {"num_vertices": 100_000, "avg_degree": 7},
    "medium": {"num_vertices": 500_000, "avg_degree": 7},
    "large": {"num_vertices": 1_000_000, "avg_degree": 7},
}

FAMILIES = ["uniform", "powerlaw_balanced", "in_heavy", "out_heavy", "multi_hub"]

# Family parameters are identical across tiers except where a value is listed
# per tier, so a family means the same generative process at every size.
FAMILY_PARAMS = {
    # k random permutations: every vertex gets out-degree k and in-degree k before cleaning.
    "uniform": {},
    # Directed Chung-Lu with one shared weight per vertex for in and out
    # (hubs are heavy on both sides). Weights ~ rank^(-1/(gamma-1)); each
    # vertex's expected edge share is capped at max_share to bound the hubs.
    "powerlaw_balanced": {"gamma": 2.5, "max_share": 0.002},
    # A hub fraction of edges point INTO hub_count randomly chosen hubs with
    # Zipf(hub_zipf_s) popularity; everything else is uniform random.
    "in_heavy": {"hub_edge_fraction": 0.25, "hubs_per_100k": 10, "hub_zipf_s": 1.0},
    # Mirror image of in_heavy: hubs have many OUTGOING edges.
    "out_heavy": {"hub_edge_fraction": 0.25, "hubs_per_100k": 10, "hub_zipf_s": 1.0},
    # hub_count hubs, each with ~hub_degree in AND out edges (+/- jitter), on a
    # uniform background. Several comparable high-work vertices.
    "multi_hub": {
        "hub_count": {"small": 10, "medium": 20, "large": 40},
        "hub_degree": 2000,
        "hub_degree_jitter": 0.25,
    },
}

# Edge policy applied identically to every synthetic graph (and reported for Pokec).
EDGE_POLICY = {
    "directed": True,
    "self_loops": "removed",
    "duplicate_edges": "removed (simple digraph: at most one src->dst edge)",
    "id_type": "int32",
    "id_range": "[0, num_vertices)",
    "top_up": "when cleaning drops edges, extra edges are sampled from the same "
              "distribution until m_target unique edges exist, then a seeded random "
              "subset of exactly m_target is kept",
}

# Hypercube planner defaults from the repo (bench/bench.py) - used to report how
# many vertices each graph would route to the hub branch.
WORK_THRESHOLD = 1_000_000

# Pre-declared topology checks. Thresholds are fixed before generation and the
# measured values are written next to them, so a family name is never the only
# evidence of a topology.
FAMILY_CHECKS = {
    "uniform": [
        ("in_cv", "<=", 0.2),
        ("out_cv", "<=", 0.2),
        ("in_max_over_mean", "<=", 2.0),
        ("out_max_over_mean", "<=", 2.0),
    ],
    "powerlaw_balanced": [
        ("in_max_over_mean", ">=", 50.0),
        ("out_max_over_mean", ">=", 50.0),
        ("in_gini", ">=", 0.4),
        ("out_gini", ">=", 0.4),
        ("log_degree_corr", ">=", 0.5),        # balanced: heavy-in vertices are heavy-out
        ("max_in_over_max_out", "between", (1 / 3, 3.0)),
    ],
    "in_heavy": [
        ("in_max_over_mean", ">=", 100.0),
        ("out_max_over_mean", "<=", 5.0),
        ("in_top_hub_edge_share", ">=", 0.15),  # share of edges entering the top-H in-degree vertices
        ("out_cv", "<=", 0.6),
    ],
    "out_heavy": [
        ("out_max_over_mean", ">=", 100.0),
        ("in_max_over_mean", "<=", 5.0),
        ("out_top_hub_edge_share", ">=", 0.15),
        ("in_cv", "<=", 0.6),
    ],
    "multi_hub": [
        ("num_work_hubs", ">=", 5),             # vertices with W > WORK_THRESHOLD
        ("top_hubs_work_share", ">=", 0.5),     # share of work_sum held by the top-H work vertices
        ("top_hubs_min_over_max_work", ">=", 0.3),  # hubs are comparable, not one dominant hub
    ],
}

# Pilot: 11 graphs spanning every family and every tier (Phase 1 asks for 8-12).
PILOT = [
    ("uniform", "small", 101),
    ("uniform", "large", 101),
    ("powerlaw_balanced", "small", 101),
    ("powerlaw_balanced", "small", 202),
    ("powerlaw_balanced", "medium", 101),
    ("in_heavy", "small", 101),
    ("in_heavy", "medium", 101),
    ("out_heavy", "small", 101),
    ("out_heavy", "large", 101),
    ("multi_hub", "small", 101),
    ("multi_hub", "medium", 101),
]
