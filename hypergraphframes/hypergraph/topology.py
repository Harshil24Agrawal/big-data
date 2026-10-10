"""
topology.py
===========
Topological profiling and degree-balanced 2D grid allocation for computational hubs.
"""

from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr


def profile_topology(
    edges: DataFrame,
    work_threshold: int = 1_000_000,
    p_cells: int = 100,
) -> Tuple[DataFrame, DataFrame]:
    """
    Profiles graph edge topology, computes Cartesian workload per intermediate vertex,
    identifies computational hubs, and allocates optimal 2D grid dimensions (r, s).

    Parameters
    ----------
    edges : DataFrame
        Spark DataFrame containing edge relations with at least 'src' and 'dst' columns.
    work_threshold : int, default=1_000_000
        Workload threshold (d_in * d_out) to classify a vertex as a heavy-hitter hub.
    p_cells : int, default=100
        Maximum grid cell capacity budget per hub (r * s <= p_cells).

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        - hub_meta: Cached DataFrame of schema (id, in_deg, out_deg, work, r_final, s_final)
        - hub_ids: Cached DataFrame containing only the 'id' column of identified hubs.
    """
    # 1. Degree Computations
    out_deg = (
        edges.groupBy("src")
        .count()
        .withColumnRenamed("src", "id")
        .withColumnRenamed("count", "out_deg")
    )
    in_deg = (
        edges.groupBy("dst")
        .count()
        .withColumnRenamed("dst", "id")
        .withColumnRenamed("count", "in_deg")
    )

    # 2. Join degrees and calculate Cartesian workload W(v) = d-(v) * d+(v)
    degrees = in_deg.join(out_deg, on="id", how="full").fillna(0)
    degrees = degrees.withColumn("work", col("in_deg") * col("out_deg"))

    # 3. Filter computational hubs exceeding work threshold
    hubs = degrees.filter(col("work") > work_threshold)

    # 4. Solve Lagrange multiplier for optimal aspect ratio:
    #    r = sqrt( (d-(h) * P) / max(d+(h), 1) )
    #    s = sqrt( (d+(h) * P) / max(d-(h), 1) )
    #    Bounded by r * s <= P
    hub_meta = (
        hubs.withColumn("raw_r", expr(f"sqrt(in_deg * {p_cells}.0 / greatest(out_deg, 1))"))
        .withColumn("raw_s", expr(f"sqrt(out_deg * {p_cells}.0 / greatest(in_deg, 1))"))
        .withColumn(
            "scale",
            expr(f"IF(raw_r * raw_s > {p_cells}, sqrt({p_cells}.0 / (raw_r * raw_s)), 1.0)"),
        )
        .withColumn("r_final", expr("cast(greatest(1, floor(raw_r * scale)) as int)"))
        .withColumn("s_final", expr("cast(greatest(1, floor(raw_s * scale)) as int)"))
        .select("id", "in_deg", "out_deg", "work", "r_final", "s_final")
        .cache()
    )

    hub_ids = hub_meta.select("id").cache()

    return hub_meta, hub_ids
