"""
core.py
=======
HyperGraphFrame wrapper for PySpark GraphFrames with topology-aware 2D partitioning.
"""

from typing import Optional, Any
from pyspark.sql import DataFrame
from hypergraph.topology import profile_topology
from hypergraph.planner import parse_2hop_motif, run_hypercube_2hop


class HyperGraphFrame:
    """
    HyperGraphFrame: A skew-resilient wrapper around PySpark GraphFrame.

    Intercepts motif .find() queries and executes degree-balanced 2D hypercube
    partitioning for heavy-hitter hub vertices to eliminate memory spills,
    task variance, and stragglers.

    Parameters
    ----------
    graph_frame : Any
        An existing GraphFrame instance or object exposing .vertices and .edges DataFrames.
    work_threshold : int, default=1_000_000
        Cartesian workload threshold (d_in * d_out) to classify a vertex as a heavy hub.
    p_cells : int, default=100
        Grid capacity budget (r * s <= p_cells) allocated per hub.
    skip_id_only_vertex_join : bool, default=False
        Opt-in optimization: when True and vertices only contains the 'id' column,
        construct vertex structs directly from edge endpoints without joining the
        vertices DataFrame. Only valid when vertices is known to contain every
        edge endpoint exactly once. Default False preserves exact native GraphFrames
        inner-join semantics (filtering dangling endpoints and preserving duplicate vertex IDs).
    """

    def __init__(
        self,
        graph_frame: Any,
        work_threshold: int = 1_000_000,
        p_cells: int = 100,
        skip_id_only_vertex_join: bool = False,
    ):
        self._gf = graph_frame
        self.work_threshold = work_threshold
        self.p_cells = p_cells
        self.skip_id_only_vertex_join = skip_id_only_vertex_join

        # Lazy Profiling cache
        self._hub_meta: Optional[DataFrame] = None
        self._hub_ids: Optional[DataFrame] = None
        self._hub_count: Optional[int] = None

    @property
    def vertices(self) -> DataFrame:
        """Returns the vertices DataFrame of the underlying GraphFrame."""
        return self._gf.vertices

    @property
    def edges(self) -> DataFrame:
        """Returns the edges DataFrame of the underlying GraphFrame."""
        return self._gf.edges

    @property
    def hub_meta(self) -> Optional[DataFrame]:
        """Returns the cached hub metadata DataFrame if profiled, else None."""
        return self._hub_meta

    @property
    def hub_ids(self) -> Optional[DataFrame]:
        """Returns the cached hub IDs DataFrame if profiled, else None."""
        return self._hub_ids

    @property
    def hub_count(self) -> Optional[int]:
        """Returns the number of identified hub vertices if profiled, else None."""
        return self._hub_count

    def profile(self) -> "HyperGraphFrame":
        """
        Manually triggers the topological profiling pass, caches metadata,
        and materializes the hub count.
        """
        if self._hub_meta is None or self._hub_ids is None or self._hub_count is None:
            # Enforce caching on base edges DataFrame so degree computation
            # doesn't re-read from source Parquet/disk multiple times
            self._gf.edges.cache()

            hub_meta, hub_ids = profile_topology(
                edges=self._gf.edges,
                work_threshold=self.work_threshold,
                p_cells=self.p_cells,
            )

            # Explicitly enforce caching on hub metadata DataFrames
            self._hub_meta = hub_meta.cache()
            self._hub_ids = hub_ids.cache()

            # Materialize the cache and capture the count
            self._hub_count = self._hub_meta.count()

        return self

    def uncache_topology(self) -> "HyperGraphFrame":
        """
        Uncaches and clears the topological hub metadata.
        """
        if self._hub_meta is not None:
            try:
                self._hub_meta.unpersist()
            except Exception:
                pass
            self._hub_meta = None

        if self._hub_ids is not None:
            try:
                self._hub_ids.unpersist()
            except Exception:
                pass
            self._hub_ids = None

        self._hub_count = None
        return self

    def find(self, pattern: str) -> DataFrame:
        """
        Finds motif patterns in the graph.

        If the motif pattern matches a 2-hop directed chain, the query is intercepted
        and executed via the 2D Hypercube Partitioning engine. For all other motifs,
        the query is transparently delegated to the native GraphFrame.find().

        Parameters
        ----------
        pattern : str
            Graph motif pattern string (e.g. '(a)-[e1]->(b); (b)-[e2]->(c)').

        Returns
        -------
        DataFrame
            Spark DataFrame containing the motif match results matching native
            GraphFrames StructType schema.
        """
        motif_spec = parse_2hop_motif(pattern)

        if motif_spec is not None:
            # Lazy Profiling: Execute topology profiling only on first skew-optimized query
            if self._hub_meta is None or self._hub_ids is None or self._hub_count is None:
                self.profile()

            return run_hypercube_2hop(
                edges=self._gf.edges,
                vertices=self._gf.vertices,
                hub_meta=self._hub_meta,
                hub_ids=self._hub_ids,
                motif_spec=motif_spec,
                hub_count=self._hub_count,
                skip_id_only_vertex_join=self.skip_id_only_vertex_join,
            )

        # Fallback to native GraphFrames.find() for unsupported motifs
        return self._gf.find(pattern)

    def __getattr__(self, name: str) -> Any:
        """
        Passes through any non-intercepted attributes or algorithms to the underlying GraphFrame.
        """
        return getattr(self._gf, name)

    def __repr__(self) -> str:
        profiled_status = (
            f"profiled (hubs={self._hub_count})"
            if self._hub_meta is not None
            else "unprofiled"
        )
        return (
            f"HyperGraphFrame(vertices={self._gf.vertices.count() if hasattr(self._gf, 'vertices') else 'N/A'}, "
            f"edges={self._gf.edges.count() if hasattr(self._gf, 'edges') else 'N/A'}, "
            f"work_threshold={self.work_threshold}, p_cells={self.p_cells}, status={profiled_status})"
        )
