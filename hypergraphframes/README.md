# HyperGraphFrames

[![PyPI version](https://img.shields.io/pypi/v/hypergraphframes.svg)](https://pypi.org/project/hypergraphframes/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hypergraphframes.svg)](https://pypi.org/project/hypergraphframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Topology-Aware Skew-Mitigating Middleware for PySpark GraphFrames.**

**HyperGraphFrames** is a Python middleware library for [Apache Spark GraphFrames](https://graphframes.github.io/graphframes/). It introduces **Topology-Aware Degree-Balanced 2D Hypercube Partitioning** for 2-hop motif joins `(a)-[e1]->(b); (b)-[e2]->(c)` on power-law graphs, flattening tail task stragglers from over $15\times$ max/median task imbalance down to about $2\times$.

---

## ⚡ The Problem: Computational & Byte Skew in Graph Joins

GraphFrames translates motif queries into relational edge-to-edge equi-joins:
$$\mathcal{M} = e_1 \bowtie_{e_1.dst = e_2.src} e_2$$

In power-law graphs, a small fraction of high-degree hub vertices creates severe execution imbalance:
1. **Computational Skew:** An intermediate hub vertex $h$ with $d^-(h)$ incoming and $d^+(h)$ outgoing edges generates a Cartesian join workload of $W(h) = d^-(h) \cdot d^+(h)$, producing millions of intermediate paths assigned to a single reducer task.
2. **Task Stragglers:** In standard SortMergeJoins, tasks processing heavy hubs become extreme stragglers, running up to $15\times$ longer than median tasks.

---

## 🚀 The Solution: 2D Hypercube Partitioning

HyperGraphFrames intercepts 2-hop motif queries before Catalyst executes them and applies a degree-balanced 2D grid partitioning strategy:
- **Lagrange Aspect Ratio Optimization:** For each heavy hub $h$ where $W(h) > \theta_w$, it computes optimal 2D grid dimensions $(r, s)$ subject to grid capacity $r \cdot s \le P$:
  $$r \approx \sqrt{\frac{d^-(h) \cdot P}{d^+(h)}}, \qquad s \approx \sqrt{\frac{d^+(h) \cdot P}{d^-(h)}}$$
- **Virtual Grid Salting:** Incoming edges ($e_1$) are hashed into $r$ row groups and replicated across $s$ columns. Outgoing edges ($e_2$) are hashed into $s$ column groups and replicated across $r$ rows.
- **3D Compound Join:** Edges are joined over the compound partition key `(dst, r_coord, s_coord)`, evenly distributing the $W(h)$ work across $P$ independent Spark tasks.
- **Asymmetric Interpolation:** For asymmetric hubs ($d^- \gg d^+$), the grid naturally degrades to 1D salting, while for symmetric hubs it achieves $O(\sqrt{P})$ replication.

---

## 📦 Installation

```bash
pip install hypergraphframes
```

> **Note on GraphFrames Dependency:**
> `hypergraphframes` requires Apache Spark (`pyspark>=3.0.0`). The underlying `graphframes` jar must be supplied via Spark's package manager when creating the `SparkSession` using the Scala coordinate matching your PySpark build (for standard PySpark 3.5.x on Scala 2.12: `--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12` or `.config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")`; for Scala 2.13 builds: `graphframes:graphframes:0.8.3-spark3.5-s_2.13`).

---

## 💡 Quickstart & Drop-in Replacement

`HyperGraphFrame` wraps your existing `GraphFrame` object:

```python
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from hypergraph import HyperGraphFrame

# 1. Initialize SparkSession with GraphFrames package (Scala 2.12 for PySpark 3.5.x)
spark = (
    SparkSession.builder.appName("HyperGraphDemo")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    .getOrCreate()
)

# 2. Create standard GraphFrame
vertices = spark.createDataFrame([("1", "Alice"), ("2", "Bob"), ("3", "Charlie")], ["id", "name"])
edges = spark.createDataFrame([("1", "2", "friend"), ("2", "3", "follow")], ["src", "dst", "rel"])
g = GraphFrame(vertices, edges)

# 3. Wrap with HyperGraphFrame (Drop-in Replacement)
hg = HyperGraphFrame(g, work_threshold=1_000_000, p_cells=100)

# 4. Execute motif query - automatically intercepted and optimized!
motifs = hg.find("(a)-[e1]->(b); (b)-[e2]->(c)")
motifs.show()
```

### Output Schema Compatibility
`HyperGraphFrame.find()` produces a DataFrame with the exact same `StructType` schema and column ordering as native GraphFrames (`[a, e1, b, e2, c]`).

---

## 🔬 Related Work

Distributed subgraph matching and motif discovery on large-scale property graphs frequently face severe workload imbalance due to power-law degree distributions. Standard relational engines, including Apache Spark, evaluate 2-hop path queries `(a)-[e1]->(b); (b)-[e2]->(c)` via relational equi-joins on intermediate vertices ($e_1.dst = e_2.src$).

1. **Multiway Joins & Worst-Case Optimal Joins (WCOJ):** Afrati and Ullman (2010) established the theoretical foundation of allocating multidimensional Cartesian share grids (the Hypercube / Shares algorithm) to minimize communication cost in distributed multiway joins. Algorithms such as SharesSkew further extended this to address key skew by isolating heavy-hitter values into dedicated sub-cubes. Systems like BiGJoin (Ammar et al., 2018) extended worst-case optimal join concepts to distributed dataflow systems (Timely Dataflow). HyperGraphFrames adapts these 2D space allocation principles specifically for 2-hop motif joins within Spark's relational DataFrame model.
2. **Degree-Aware Relational Salting:** Standard 1D salting distributes skewed keys across $K$ buckets on the probe side while replicating the build side across all $K$ buckets. In directed motifs, heavy intermediate hubs often exhibit asymmetric in/out degrees ($d_{in} \neq d_{out}$). HyperGraphFrames profiles directed degrees to compute an optimal 2D aspect ratio ($r \times s \le P$) that balances replication and partitioning across both edge inputs.
3. **GraphFrames Motif Compilation:** GraphFrames (Dave et al., 2016) compiles motif queries into sequences of DataFrame inner joins. GraphFrames typically joins vertex attributes to edges before evaluating the motif expansion. However, when intermediate vertices have large directed degree products ($d_{in} \cdot d_{out}$), standard SortMergeJoins must buffer large same-key groups, resulting in task stragglers.

---

## 📊 Controlled Empirical Evaluation

All benchmarks were conducted using a controlled, reproducible harness across $N=5$ randomized, interleaved runs per condition on the Stanford SNAP **`soc-pokec`** directed social network (30,622,564 directed edges, 1,632,803 vertices), producing 1,780,294,899 materialized paths for the 2-hop motif `(a)-[e1]->(b); (b)-[e2]->(c)`.

### Benchmark Environment & Methodology
* **Host Environment:** Single-node workstation execution on AMD Ryzen AI 7 350 w/ Radeon 860M (8 cores / 16 threads), 16 GB RAM, Micron 512 GB NVMe SSD (Windows 11).
* **Runtime:** Apache Spark 3.5.5 on Eclipse Temurin OpenJDK 17.0.20.1 (Scala 2.12), Python 3.11.
* **Spark Configuration:** `local[*]`, driver memory 6 GB, 200 shuffle partitions, `spark.sql.autoBroadcastJoinThreshold = -1`.
  * *Note on Execution Mode*: In `local[*]` mode, all executor tasks run in-process within a single JVM on a single machine, sharing memory and CPU cores.
* **Materialization:** Action executed via `write.format("noop").mode("overwrite").save()` to ensure all 5 output structs are computed without benchmarking disk sink bottlenecks.
* **Vertex Broadcasting in Benchmark:** To isolate the edge-to-edge join while avoiding secondary shuffle bottlenecks, the cached vertex lookup table is broadcast-hinted (`broadcast(vertices)`) uniformly across all variants in the benchmark harness.

### Full 5-Run Benchmark Results

#### Action: `noop` | Regime: `id_only`
| Variant | Role | AQE | N | Query Latency (s)<br>median [min–max] (IQR) | Total Latency (incl. prof.) (s)<br>median [min–max] | Dominant Stage<br>Max/Median Ratio | Executor Run Time (excl. prof.) (s)<br>median | Shuffle Read/Write | Memory Spill | Disk Spill |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **native** | Reference (GraphFrames) | off | 5 | `62.31 [52.51–67.72] (IQR 8.35)` | `62.31 [52.51–67.72] (IQR 8.35)` | `14.64 [max 15.75]` | `775.0` | 1.07 GB | 0 MB | 0 MB |
| **edgejoin** | Baseline 1 (unpartitioned) | off | 5 | `72.17 [70.52–77.86] (IQR 4.99)` | `72.17 [70.52–77.86] (IQR 4.99)` | `11.81 [max 12.24]` | `1059.7` | 675.4 MB | 0 MB | 0 MB |
| **salt1d** | Baseline 2 (1D salting) | off | 5 | `88.21 [82.94–92.14] (IQR 6.00)` | `106.92 [99.22–110.52] (IQR 7.91)` | `1.98 [max 2.08]` | `1356.6` | 716.5 MB | 0 MB | 0 MB |
| **hypercube** | 2D Partitioned (opt-in) | off | 5 | `81.53 [81.19–90.14] (IQR 7.46)` | `97.55 [97.07–108.73] (IQR 9.80)` | `2.21 [max 2.25]` | `1257.5` | 688.6 MB | 0 MB | 0 MB |
| **hypercube_vjoin** | Drop-in Default (vjoin=on) | off | 5 | `86.94 [77.29–88.99] (IQR 10.00)` | `104.51 [92.82–107.94] (IQR 13.37)` | `1.85 [max 2.07]` | `1327.0` | 688.6 MB | 0 MB | 0 MB |
| **native** | Reference (GraphFrames) | on | 5 | `61.50 [57.94–75.38] (IQR 12.28)` | `61.50 [57.94–75.38] (IQR 12.28)` | `1.51 [max 1.76]` | `745.4` | 1.07 GB | 3.09 GB | 613.5 MB |
| **edgejoin** | Baseline 1 (unpartitioned) | on | 5 | `82.32 [72.83–84.02] (IQR 7.62)` | `82.32 [72.83–84.02] (IQR 7.62)` | `1.32 [max 1.37]` | `1071.2` | 675.4 MB | 1.75 GB | 216.2 MB |
| **salt1d** | Baseline 2 (1D salting) | on | 5 | `95.89 [87.79–101.68] (IQR 13.53)` | `115.98 [104.12–120.71] (IQR 15.55)` | `1.31 [max 1.37]` | `1248.7` | 716.5 MB | 1.74 GB | 235.2 MB |
| **hypercube** | 2D Partitioned (opt-in) | on | 5 | `95.00 [86.12–100.58] (IQR 9.47)` | `113.62 [103.00–118.56] (IQR 9.95)` | `1.32 [max 1.36]` | `1251.3` | 688.6 MB | 1.64 GB | 202.5 MB |
| **hypercube_vjoin** | Drop-in Default (vjoin=on) | on | 5 | `108.79 [95.76–116.63] (IQR 14.12)` | `126.68 [112.68–137.44] (IQR 15.82)` | `1.66 [max 1.70]` | `1272.0` | 688.6 MB | 1.64 GB | 202.5 MB |

#### Action: `noop` | Regime: `with_attrs`
| Variant | Role | AQE | N | Query Latency (s)<br>median [min–max] (IQR) | Total Latency (incl. prof.) (s)<br>median [min–max] | Dominant Stage<br>Max/Median Ratio | Executor Run Time (excl. prof.) (s)<br>median | Shuffle Read/Write | Memory Spill | Disk Spill |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **native** | Reference (GraphFrames) | off | 5 | `64.79 [56.98–78.02] (IQR 12.54)` | `64.79 [56.98–78.02] (IQR 12.54)` | `14.39 [max 17.00]` | `827.5` | 1.19 GB | 0 MB | 0 MB |
| **edgejoin** | Baseline 1 (unpartitioned) | off | 5 | `99.64 [87.76–102.37] (IQR 11.19)` | `99.64 [87.76–102.37] (IQR 11.19)` | `14.80 [max 15.89]` | `1289.2` | 675.4 MB | 0 MB | 0 MB |
| **salt1d** | Baseline 2 (1D salting) | off | 5 | `89.41 [82.81–91.57] (IQR 7.82)` | `106.15 [98.40–109.99] (IQR 10.33)` | `1.85 [max 1.91]` | `1363.0` | 716.5 MB | 0 MB | 0 MB |
| **hypercube** | 2D Partitioned | off | 5 | `83.70 [82.30–93.06] (IQR 8.28)` | `101.90 [97.99–111.83] (IQR 11.05)` | `1.87 [max 2.10]` | `1284.6` | 688.6 MB | 0 MB | 0 MB |
| **native** | Reference (GraphFrames) | on | 5 | `78.76 [60.17–87.55] (IQR 24.20)` | `78.76 [60.17–87.55] (IQR 24.20)` | `1.77 [max 1.92]` | `817.5` | 1.19 GB | 3.34 GB | 658.7 MB |
| **edgejoin** | Baseline 1 (unpartitioned) | on | 5 | `106.18 [92.69–118.49] (IQR 19.52)` | `106.18 [92.69–118.49] (IQR 19.52)` | `1.54 [max 1.57]` | `1258.8` | 675.4 MB | 1.84 GB | 225.3 MB |
| **salt1d** | Baseline 2 (1D salting) | on | 5 | `131.44 [110.19–146.39] (IQR 24.31)` | `151.38 [126.65–166.43] (IQR 27.59)` | `1.82 [max 1.95]` | `1366.5` | 716.5 MB | 1.74 GB | 235.2 MB |
| **hypercube** | 2D Partitioned | on | 5 | `127.70 [122.52–134.47] (IQR 9.91)` | `147.34 [142.37–154.91] (IQR 9.84)` | `1.77 [max 1.84]` | `1373.0` | 688.6 MB | 1.84 GB | 225.1 MB |

---

## 🔍 Empirical Analysis & Limitations

1. **Task Skew Mitigation vs. Mean Task Overhead:**
   Under AQE off, standard unpartitioned joins (`edgejoin` and `native`) exhibit severe task stragglers, with maximum-to-median task duration ratios ranging from **$11.8\times$ to $15.8\times$** (individual tasks running 45–60s while the median task completes in under 4s). 2D hypercube partitioning successfully reduces this ratio to **about $2\times$** ($1.85\times$–$2.21\times$ median, $2.25\times$ max). However, achieving this balance requires replicating edges into grid cells, which increases aggregate task count and total executor CPU time.

2. **Hypercube Ties 1D Salting (`salt1d`):**
   Across both query latency and total wall-clock time, 2D hypercube partitioning performs essentially on par with 1D salting (`salt1d`) without demonstrating meaningful runtime separation (e.g., in `id_only` AQE-off, total latency is 97.6s for hypercube vs. 106.9s for salt1d; under AQE-on, 113.6s vs. 116.0s; in `with_attrs` AQE-on, 147.3s vs. 151.4s). Both approaches achieve comparable straggler reduction (~1.85×–2.2× without AQE), showing that 2D grid allocation provides minimal wall-clock advantage over 1D salting on this workload.

3. **Unpartitioned Joins Are as Fast or Faster on Single-Node:**
   On a single-node workstation, unpartitioned joins (`edgejoin` and GraphFrames `native`) are as fast or faster in total wall-clock time across all evaluated conditions. Total wall-clock time for hypercube is slower than unpartitioned joins across all conditions, and query latency alone is slower in three of four conditions (beating `edgejoin` query latency only in `with_attrs` AQE-off: 83.7s vs. 99.6s, but still slower in total latency at 101.9s vs. 99.6s). GraphFrames `native` remains the fastest overall across all benchmarks (61–65s total latency), avoiding both the degree profiling pass (~16–18s) and the additional shuffle stages of partitioned execution.

4. **AQE Flattens Task Stragglers Automatically:**
   When Spark's Adaptive Query Execution (AQE) is enabled, Spark dynamically mitigates partition skew at runtime, reducing the dominant stage max/median ratio to ~1.3×–1.8× across all variants (for unpartitioned `edgejoin`, AQE flattens the ratio from 11.8× down to 1.32×). AQE thus neutralizes the straggler mitigation advantage of application-level partitioning. Furthermore, under single-node memory pressure, AQE partition coalescing triggers 1.6–1.8 GB memory spill and >200 MB disk spill for custom variants (and 3.1 GB memory spill / 613 MB disk spill on native), increasing query duration.

5. **Single-Node vs. Multi-Node Cluster Tradeoffs (Hypothesis):**
   In this single-node `local[*]` workstation benchmark, communication cost is bounded by local memory bandwidth, and the degree profiling pass (~16–18 seconds) represents a measurable fraction of total query wall-clock time. While hypercube partitioning successfully reduces task stragglers, total wall-clock time is slower than unpartitioned joins across conditions on a single multi-core machine. *We hypothesize that in distributed multi-node clusters—where stragglers block cluster-wide stage synchronization barriers and network shuffle interconnects are bottlenecked—2D hypercube partitioning could yield net end-to-end speedups.* Testing this hypothesis across multi-node infrastructure remains an area for future work.

6. **Vertex Join Sequencing:**
   Custom motif joining evaluates the 2-hop edge expansion before joining vertex properties. When vertex properties cannot be broadcast or contain large payloads, joining attributes post-expansion increases row processing volume relative to planners that join vertex attributes directly to incident edges prior to path expansion.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
