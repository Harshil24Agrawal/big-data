# Topology-Aware Skew Handling for GraphFrames Motif Queries

## 1. The Fundamental Problem
GraphFrames provides a convenient DataFrame-based interface for graph analytics in Apache Spark, enabling motif queries to be integrated with relational data processing pipelines. However, GraphFrames translates graph motifs (e.g., `(a)-[]->(b); (b)-[]->(c)`) directly into relational edge-to-edge joins (`e1.join(e2, e1.dst == e2.src)`).

The reliance on standard relational execution creates two distinct categories of data skew on power-law graphs:
1. **Byte Skew:** A hub vertex has so many incident edges that its shuffle partition dominates network transfer and disk I/O.
2. **Computational Skew:** The motif join workload for an intermediate vertex $h$ is defined by its directed degree product: $W(h) = d^{-}(h)d^{+}(h)$. A vertex can have modest input bytes (e.g., 4,000 incoming and 4,000 outgoing edges equal roughly 800 KB) but produce a catastrophic Cartesian output explosion (16 million motif paths). 

In Spark’s inner `SortMergeJoin`, one input is streamed while rows with the matching key from the other input are buffered. Spark stores these rows in an `ExternalAppendOnlyUnsafeRowArray`, which may transition to an `UnsafeExternalSorter` and spill to disk. High-fanout hub keys create massive buffered same-key row groups, leading to severe task stragglers, out-of-memory errors, and disk spill.

## 2. The Limitations of the Existing Approach (Spark AQE)
Apache Spark 3.x features Adaptive Query Execution (AQE), which dynamically optimizes execution plans using runtime statistics. AQE’s skew-join optimization can detect a skewed partition and split it to mitigate stragglers.

However, AQE is generic and threshold-driven. By default, AQE classifies a partition as skewed only if it exceeds **both** an absolute byte threshold (default: 256 MB) **and** a relative threshold (default: 5× the median partition size). 
Consequently, computational hubs that generate millions of output paths but have a raw input size below the byte threshold are entirely ignored by AQE. Furthermore, because AQE operates reactively post-shuffle, it cannot prevent the map-side funneling of a massive hub's edges into a single logical partition.

## 3. The Proposed Approach: Degree-Balanced Hypercube Partitioning
We propose a topology-aware skew-handling planner that leverages structural graph statistics before the motif edge join is executed. Drawing upon SharesSkew and hypercube partitioning literature, our planner applies a specialized vertex-cut strategy natively within the Spark DataFrame API.

For two-way edge joins with a heavy intermediate-vertex key, the planner calculates optimal 2D grid dimensions $(r, s)$ to minimize the communication objective $C(h) = d^{-}(h)s + d^{+}(h)r$, subject to $r \cdot s \le P$ (where $P$ is the number of grid cells).

$$ r \approx \sqrt{\frac{d^{-}(h)P}{d^{+}(h)}}, \qquad s \approx \sqrt{\frac{d^{+}(h)P}{d^{-}(h)}} $$

Based on this degree imbalance, the planner continuously interpolates between a 2D shares partition and its 1D limiting case. For example, an asymmetric hub with $d^- = 10,000$ and $d^+ = 100$ evaluated over $P=100$ cells yields $r=100, s=1$, naturally degrading to a 1D salt where outgoing edges are replicated 100 times and incoming edges are not replicated.

To execute this within Catalyst, the planner identifies high-work computational hubs and routes their edges to a dedicated execution branch:
- Incoming edges are hashed into $r$ row groups and replicated across $s$ columns.
- Outgoing edges are hashed into $s$ column groups and replicated across $r$ rows.
- The motif join completes at the intersection of these grid coordinates.

## 4. How it Solves the Problem
This framework tackles both computational and byte skew proactively:
1. **Reduction in Replicated Input Traffic:** For a symmetric hub, replicating both relations across a $\sqrt{P} \times \sqrt{P}$ grid reduces total shuffled input traffic to $O(\sqrt{P})$ per relation, compared to $O(P)$ under 1D salting. For $P=100$, expanded-input traffic drops by approximately $5.05\times$.
2. **Bounding the Cartesian Workload:** The grid divides the $d^{-}(h)d^{+}(h)$ result cardinality across $P$ disjoint tasks. This reduces the number of hub-key rows assigned to each grid cell, significantly lowering the per-task `ExternalAppendOnlyUnsafeRowArray` buffer pressure, join work, and spill risk.

## 5. Scope and Empirical Benchmark Design
This optimization aims to reduce task variance and spill risk for foundational 2-hop motif joins. We explicitly uncouple base-topology statistics from Catalyst’s cost-based predicate estimations, acknowledging that highly selective edge filters may alter effective runtime degrees.

To validate the hypothesis that prior topology knowledge outperforms generic byte-threshold heuristics, we evaluate the planner across three synthetic structures:
1. **Uniform-Degree Graph (Control):** To quantify the degree-precomputation and branching overhead.
2. **Byte-Skewed Hub:** A 5 MB heavy-key partition configured against a tiny median partition (e.g., < 200 KB) with `skewedPartitionThresholdInBytes=1MB`, ensuring AQE has a fair opportunity to trigger.
3. **Compute-Skewed Hub:** A hub with massive $d^{-}d^{+}$ output but input bytes strictly below AQE detection thresholds.

Metrics gathered via the Spark REST API include maximum and p95 join-stage task duration, task-duration coefficient of variation (CV), shuffle bytes/records, spill bytes, and end-to-end runtime.
