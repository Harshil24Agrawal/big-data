"""
planner.py
==========
Motif pattern parsing and 2D Hypercube execution engine for PySpark GraphFrames.
"""

import re
from typing import Optional, Dict, Any, List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, expr, explode, sequence, broadcast, struct


def parse_2hop_motif(motif_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses a GraphFrames motif string to check if it represents a 2-hop directed chain.
    Supported patterns:
        - (a)-[e1]->(b); (b)-[e2]->(c)
        - (a)-[]->(b); (b)-[]->(c)
        - (b)-[e2]->(c); (a)-[e1]->(b)
        - (a)-[e1]->(b); (b)-[e2]->(a) (2-cycle)

    Parameters
    ----------
    motif_str : str
        The motif pattern string passed to .find().

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing motif specification and token ordering if matched,
        otherwise None.
    """
    if not isinstance(motif_str, str):
        return None

    clause_regex = r"\(\s*([a-zA-Z_0-9]+)\s*\)\s*-\s*\[\s*([a-zA-Z_0-9]*)\s*\]\s*->\s*\(\s*([a-zA-Z_0-9]+)\s*\)"
    clauses = [c.strip() for c in motif_str.split(";") if c.strip()]
    if len(clauses) != 2:
        return None

    m1 = re.fullmatch(clause_regex, clauses[0])
    m2 = re.fullmatch(clause_regex, clauses[1])
    if not m1 or not m2:
        return None

    src1, e1, dst1 = m1.group(1), m1.group(2) or None, m1.group(3)
    src2, e2, dst2 = m2.group(1), m2.group(2) or None, m2.group(3)

    # Preserve token ordering as declared in the input query
    ordered_tokens: List[str] = []
    for m in [m1, m2]:
        u, e, v = m.group(1), m.group(2) or None, m.group(3)
        if u not in ordered_tokens:
            ordered_tokens.append(u)
        if e and e not in ordered_tokens:
            ordered_tokens.append(e)
        if v not in ordered_tokens:
            ordered_tokens.append(v)

    if dst1 == src2:
        return {
            "a": src1,
            "e1": e1,
            "b": dst1,
            "e2": e2,
            "c": dst2,
            "ordered_tokens": ordered_tokens,
        }
    elif dst2 == src1:
        return {
            "a": src2,
            "e1": e2,
            "b": dst2,
            "e2": e1,
            "c": dst1,
            "ordered_tokens": ordered_tokens,
        }

    return None


def run_hypercube_2hop(
    edges: DataFrame,
    vertices: Optional[DataFrame],
    hub_meta: DataFrame,
    hub_ids: DataFrame,
    motif_spec: Dict[str, Any],
    hub_count: int,
) -> DataFrame:
    """
    Executes the 2-hop motif query using Topology-Aware Degree-Balanced 2D Hypercube Partitioning.

    Parameters
    ----------
    edges : DataFrame
        The graph edges DataFrame (must have 'src' and 'dst').
    vertices : Optional[DataFrame]
        The graph vertices DataFrame (must have 'id').
    hub_meta : DataFrame
        Cached metadata for heavy hubs with optimal grid dimensions (r_final, s_final).
    hub_ids : DataFrame
        Cached DataFrame with hub IDs for fast broadcast filtering.
    motif_spec : Dict[str, Any]
        Parsed motif specification containing aliases and output token sequence.
    hub_count : int
        Pre-computed number of identified hubs (avoids eager .take(1) during planning).

    Returns
    -------
    DataFrame
        A standard Spark DataFrame formatted with exact GraphFrames StructType schema.
    """
    e_cols = edges.columns
    v_cols = vertices.columns if vertices is not None else []
    pair_cols = [f"__e1_{c}" for c in e_cols] + [f"__e2_{c}" for c in e_cols]

    if hub_count == 0:
        # Fast path: No hubs present -> standard equi-join
        e1_proj = [col(c).alias(f"__e1_{c}") for c in e_cols]
        e2_proj = [col(c).alias(f"__e2_{c}") for c in e_cols]
        e1 = edges.select(*e1_proj).alias("e1")
        e2 = edges.select(*e2_proj).alias("e2")
        cond = col("e1.__e1_dst") == col("e2.__e2_src")
        if motif_spec["a"] == motif_spec["c"]:
            cond = cond & (col("e1.__e1_src") == col("e2.__e2_dst"))
        matched_edges = e1.join(e2, cond).select(
            *[col(f"e1.{c}").alias(c) if c.startswith("__e1_") else col(f"e2.{c}").alias(c) for c in pair_cols]
        )
    else:
        # ------------------ BRANCH 1: HUB 2D GRID JOIN ------------------
        # e1 incoming edges (dst is hub): hash src into r groups, replicate across s columns
        e1_hub = (
            edges.join(broadcast(hub_meta), edges["dst"] == hub_meta["id"])
            .select(
                *[edges[c].alias(f"__e1_{c}") for c in e_cols],
                hub_meta["r_final"].alias("__r"),
                hub_meta["s_final"].alias("__s"),
            )
            .withColumn("__r_coord", expr("pmod(hash(__e1_src), __r)"))
            .withColumn("__s_coord", explode(sequence(lit(0), col("__s") - 1)))
            .select(*[f"__e1_{c}" for c in e_cols], "__r_coord", "__s_coord")
            .alias("e1h")
        )

        # e2 outgoing edges (src is hub): hash dst into s groups, replicate across r rows
        e2_hub = (
            edges.join(broadcast(hub_meta), edges["src"] == hub_meta["id"])
            .select(
                *[edges[c].alias(f"__e2_{c}") for c in e_cols],
                hub_meta["r_final"].alias("__r"),
                hub_meta["s_final"].alias("__s"),
            )
            .withColumn("__s_coord", expr("pmod(hash(__e2_dst), __s)"))
            .withColumn("__r_coord", explode(sequence(lit(0), col("__r") - 1)))
            .select(*[f"__e2_{c}" for c in e_cols], "__r_coord", "__s_coord")
            .alias("e2h")
        )

        hub_cond = (
            (col("e1h.__e1_dst") == col("e2h.__e2_src"))
            & (col("e1h.__r_coord") == col("e2h.__r_coord"))
            & (col("e1h.__s_coord") == col("e2h.__s_coord"))
        )
        if motif_spec["a"] == motif_spec["c"]:
            hub_cond = hub_cond & (col("e1h.__e1_src") == col("e2h.__e2_dst"))

        hub_join = e1_hub.join(e2_hub, hub_cond).select(
            *[col(f"e1h.{c}").alias(c) if c.startswith("__e1_") else col(f"e2h.{c}").alias(c) for c in pair_cols]
        )

        # ------------------ BRANCH 2: NON-HUB HASH JOIN -----------------
        e1_proj = [col(c).alias(f"__e1_{c}") for c in e_cols]
        e2_proj = [col(c).alias(f"__e2_{c}") for c in e_cols]

        e1_nonhub = (
            edges.join(broadcast(hub_ids), edges["dst"] == hub_ids["id"], "left_anti")
            .select(*e1_proj)
            .alias("e1n")
        )
        e2_nonhub = (
            edges.join(broadcast(hub_ids), edges["src"] == hub_ids["id"], "left_anti")
            .select(*e2_proj)
            .alias("e2n")
        )

        nonhub_cond = col("e1n.__e1_dst") == col("e2n.__e2_src")
        if motif_spec["a"] == motif_spec["c"]:
            nonhub_cond = nonhub_cond & (col("e1n.__e1_src") == col("e2n.__e2_dst"))

        nonhub_join = e1_nonhub.join(e2_nonhub, nonhub_cond).select(
            *[col(f"e1n.{c}").alias(c) if c.startswith("__e1_") else col(f"e2n.{c}").alias(c) for c in pair_cols]
        )

        # ------------------ BRANCH 3: UNION RESULTS ---------------------
        matched_edges = hub_join.unionByName(nonhub_join)

    # ------------------ OUTPUT SHAPING (STRUCTS) ------------------------
    if vertices is not None:
        # Optimization: If vertices only contains 'id', we directly project from the joined edge keys
        # avoiding 3 redundant multi-million/billion-row shuffle joins with the vertex table.
        if set(v_cols) == {"id"}:
            struct_map: Dict[str, Any] = {
                motif_spec["a"]: struct(col("__e1_src").alias("id")).alias(motif_spec["a"]),
                motif_spec["b"]: struct(col("__e1_dst").alias("id")).alias(motif_spec["b"]),
            }
            if motif_spec["a"] != motif_spec["c"]:
                struct_map[motif_spec["c"]] = struct(col("__e2_dst").alias("id")).alias(motif_spec["c"])

            if motif_spec["e1"]:
                struct_map[motif_spec["e1"]] = struct(*[col(f"__e1_{c}").alias(c) for c in e_cols]).alias(motif_spec["e1"])
            if motif_spec["e2"]:
                struct_map[motif_spec["e2"]] = struct(*[col(f"__e2_{c}").alias(c) for c in e_cols]).alias(motif_spec["e2"])

            final_cols = [struct_map[tok] for tok in motif_spec["ordered_tokens"]]
            return matched_edges.select(*final_cols)
        else:
            # Full vertex attribute join when extra properties are present (name, age, etc.)
            va = vertices.select(*[col(c).alias(f"__va_{c}") for c in v_cols])
            vb = vertices.select(*[col(c).alias(f"__vb_{c}") for c in v_cols])
            vc = vertices.select(*[col(c).alias(f"__vc_{c}") for c in v_cols])

            joined = (
                matched_edges
                .join(va, col("__va_id") == col("__e1_src"))
                .join(vb, col("__vb_id") == col("__e1_dst"))
            )
            if motif_spec["a"] != motif_spec["c"]:
                joined = joined.join(vc, col("__vc_id") == col("__e2_dst"))

            struct_map = {
                motif_spec["a"]: struct(*[col(f"__va_{c}").alias(c) for c in v_cols]).alias(motif_spec["a"]),
                motif_spec["b"]: struct(*[col(f"__vb_{c}").alias(c) for c in v_cols]).alias(motif_spec["b"]),
            }
            if motif_spec["a"] != motif_spec["c"]:
                struct_map[motif_spec["c"]] = struct(*[col(f"__vc_{c}").alias(c) for c in v_cols]).alias(motif_spec["c"])

            if motif_spec["e1"]:
                struct_map[motif_spec["e1"]] = struct(*[col(f"__e1_{c}").alias(c) for c in e_cols]).alias(motif_spec["e1"])
            if motif_spec["e2"]:
                struct_map[motif_spec["e2"]] = struct(*[col(f"__e2_{c}").alias(c) for c in e_cols]).alias(motif_spec["e2"])

            final_cols = [struct_map[tok] for tok in motif_spec["ordered_tokens"]]
            return joined.select(*final_cols)
    else:
        return matched_edges
