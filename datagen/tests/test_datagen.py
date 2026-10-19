"""
Fast (numpy-only) tests for the Phase 1 generators and validators.
Run from the repo root:  python -m pytest datagen/tests -q
"""

import copy
import csv
import itertools
import os
import random

import numpy as np
import pytest

from datagen import config
from datagen.generate import MANIFEST_COLUMNS, edges_sha256, graph_id_for, spec_for
from datagen.generators import clean_edges, generate
from datagen.validate import degree_stats, run_family_checks, structural_checks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_clean_edges_removes_self_loops_and_duplicates():
    src = np.array([1, 1, 2, 3, 3, 3], dtype=np.int32)
    dst = np.array([1, 2, 3, 4, 4, 1], dtype=np.int32)
    keys, info = clean_edges(src, dst)
    assert len(keys) == 4                      # (1,2) (2,3) (3,4) (3,1)
    assert info["self_loops_removed"] == 1
    assert info["duplicates_removed"] == 1


@pytest.mark.parametrize("family", config.FAMILIES)
def test_generation_is_deterministic_per_seed(family):
    (s1, d1), _ = generate(family, "small", 101)
    (s2, d2), _ = generate(family, "small", 101)
    (s3, d3), _ = generate(family, "small", 202)
    assert edges_sha256(s1, d1) == edges_sha256(s2, d2)
    assert edges_sha256(s1, d1) != edges_sha256(s3, d3)


@pytest.mark.parametrize("family", config.FAMILIES)
def test_small_tier_passes_family_checks(family):
    (src, dst), info = generate(family, "small", 101)
    n = config.TIERS["small"]["num_vertices"]
    hub_k = int(info["family_meta"].get("hub_count", 10))
    stats = degree_stats(src, dst, n, hub_k=hub_k)
    structure = structural_checks(src, dst, n)
    status, checks = run_family_checks(family, stats, structure)
    assert status == "PASS", [c for c in checks if not c["passed"]]
    assert structure["self_loops"] == 0 and structure["duplicate_edges"] == 0 and structure["ids_in_range"]
    assert src.dtype == np.int32 and dst.dtype == np.int32
    # Near the target size (multi_hub may lose a few background edges that duplicate hub edges).
    m_target = n * config.TIERS["small"]["avg_degree"]
    assert abs(len(src) - m_target) <= 0.001 * m_target


def test_family_checks_reject_the_wrong_topology():
    """A uniform graph must not pass the power-law or multi-hub checks (family names are not proof)."""
    (src, dst), _ = generate("uniform", "small", 101)
    n = config.TIERS["small"]["num_vertices"]
    stats = degree_stats(src, dst, n)
    structure = structural_checks(src, dst, n)
    for fam in ("powerlaw_balanced", "in_heavy", "out_heavy", "multi_hub"):
        status, _ = run_family_checks(fam, stats, structure)
        assert status == "FAIL", fam


def test_work_sum_equals_brute_force_two_hop_count():
    rng = random.Random(7)
    edges = {(rng.randrange(40), rng.randrange(40)) for _ in range(300)}
    edges = {(a, b) for a, b in edges if a != b}
    src = np.array([a for a, _ in edges], dtype=np.int32)
    dst = np.array([b for _, b in edges], dtype=np.int32)
    brute = sum(1 for (a, b), (c, d) in itertools.product(edges, edges) if b == c)
    assert degree_stats(src, dst, 40)["work_sum"] == brute


def test_graph_id_is_stable_and_parameter_sensitive(monkeypatch):
    spec = spec_for("powerlaw_balanced", "small", 101)
    assert graph_id_for(spec) == graph_id_for(spec_for("powerlaw_balanced", "small", 101))
    changed = copy.deepcopy(spec)
    changed["params"]["gamma"] = 2.6
    assert graph_id_for(changed) != graph_id_for(spec)
    assert graph_id_for(spec).startswith("powerlaw_balanced-small-s101-")


def test_manifest_has_required_columns():
    required = {"graph_id", "family", "seed", "size_tier", "num_vertices", "num_edges", "generator_params",
                "path", "format", "generation_seconds", "validation_status"}
    assert required <= set(MANIFEST_COLUMNS)
    path = os.path.join(REPO_ROOT, "datasets", "manifest", "dataset_manifest.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows and required <= set(rows[0])
        assert len({r["graph_id"] for r in rows}) == len(rows)
