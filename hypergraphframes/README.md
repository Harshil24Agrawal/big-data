# HyperGraphFrames

[![PyPI version](https://img.shields.io/pypi/v/hypergraphframes.svg)](https://pypi.org/project/hypergraphframes/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hypergraphframes.svg)](https://pypi.org/project/hypergraphframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Topology-Aware Skew Handling Middleware for PySpark GraphFrames.**

**HyperGraphFrames** is a high-performance, skew-resilient Python middleware library for [Apache Spark GraphFrames](https://graphframes.github.io/graphframes/). It introduces **Topology-Aware Degree-Balanced 2D Hypercube Partitioning** for motif discovery queries on power-law graphs, completely eliminating join skew, `ExternalAppendOnlyUnsafeRowArray` buffer memory spills, and reducer stragglers.

---

## ⚡ The Problem: Computational & Byte Skew in Graph Joins

GraphFrames translates motif queries (such as 2-hop paths `(a)-[e1]->(b); (b)-[e2]->(c)`) into relational edge-to-edge equi-joins:
$$\mathcal{M} = e_1 \bowtie_{e_1.dst = e_2.src} e_2$$

In real-world power-law graphs (social networks, web graphs, citation networks), a small fraction of high-degree hub vertices creates catastrophic join skew:
1. **Computational Skew:** An intermediate hub vertex $h$ with $d^-(h)$ incoming and $d^+(h)$ outgoing edges generates a Cartesian workload of $W(h) = d^-(h) \cdot d^+(h)$. Even if the raw input size is modest (e.g., < 1 MB) and evades Spark Adaptive Query Execution (AQE) skew detection, it produces an output explosion (millions of paths) assigned to a single reducer task.
2. **Buffer Spills & OOM:** Spark's `SortMergeJoin` buffers matching same-key rows into an `ExternalAppendOnlyUnsafeRowArray`, causing multi-gigabyte memory and disk spills, severe task duration variance, and stragglers.

---

## 🚀 The Solution: 2D Hypercube Partitioning

HyperGraphFrames intercepts motif queries before Catalyst executes them and applies an optimal vertex-cut strategy:
- **Lagrange Aspect Ratio Optimization:** For each heavy hub $h$ where $W(h) > \theta_w$, it solves the optimal 2D grid dimensions $(r, s)$ subject to grid capacity $r \cdot s \le P$:
  $$r \approx \sqrt{\frac{d^-(h) \cdot P}{d^+(h)}}, \qquad s \approx \sqrt{\frac{d^+(h) \cdot P}{d^-(h)}}$$
- **Virtual Grid Salting:** Incoming edges ($e_1$) are hashed into $r$ row groups and replicated across $s$ columns. Outgoing edges ($e_2$) are hashed into $s$ column groups and replicated across $r$ rows.
- **3D Compound Join:** Edges are joined over the compound partition key `(dst, r_coord, s_coord)`, evenly distributing the $W(h)$ work across $P$ independent Spark tasks.
- **Asymmetric Interpolation:** For asymmetric hubs ($d^- \gg d^+$), the grid naturally degrades to 1D salting, while for symmetric hubs it achieves $O(\sqrt{P})$ replication ($5.05\times$ less data replication than 1D salting at $P=100$).

---

## 📦 Installation

```bash
pip install hypergraphframes
```

> **Note on GraphFrames Dependency:**
> `hypergraphframes` requires Apache Spark (`pyspark>=3.0.0`). The underlying `graphframes` library must be supplied via Spark's package manager when creating the `SparkSession` (e.g., `--packages graphframes:graphframes:0.8.3-spark3.5-s_2.13` or `.config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")`).

---

## 💡 Quickstart & Drop-in Replacement

`HyperGraphFrame` is a seamless wrapper around native `GraphFrame`. Simply wrap your existing `GraphFrame` object:

```python
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from hypergraph import HyperGraphFrame

# 1. Initialize SparkSession with GraphFrames package
spark = (
    SparkSession.builder.appName("HyperGraphDemo")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
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
`HyperGraphFrame.find()` produces a DataFrame with the exact same `StructType` schema and column ordering as native GraphFrames:

```
root
 |-- a: struct (nullable = false)
 |    |-- id: string
 |    |-- name: string
 |-- e1: struct (nullable = false)
 |    |-- src: string
 |    |-- dst: string
 |    |-- rel: string
 |-- b: struct (nullable = false)
 |    |-- id: string
 |    |-- name: string
 |-- e2: struct (nullable = false)
 |    |-- src: string
 |    |-- dst: string
 |    |-- rel: string
 |-- c: struct (nullable = false)
 |    |-- id: string
 |    |-- name: string
```

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `work_threshold` | `int` | `1_000_000` | Cartesian workload threshold $W(v) = d^-(v) \cdot d^+(v)$ to classify vertex $v$ as a heavy computational hub. |
| `p_cells` | `int` | `100` | Total virtual grid cell budget per hub ($r \times s \le P$). |

### Lazy Profiling
`HyperGraphFrame` utilizes lazy profiling: the graph topology profiling pass runs only when the first skew-optimized query is evaluated, caching hub metadata for subsequent queries. You can also trigger profiling manually or clear it:

```python
hg.profile()           # Explicitly compute and cache hub metadata
hg.uncache_topology()  # Clear cached topology metadata
```

### Transparent Fallback
Any motif pattern not currently handled by the 2D hypercube engine (e.g., 3-hop motifs or complex subgraphs) is automatically and transparently routed to native `GraphFrame.find()`, ensuring 100% backward compatibility with existing pipelines.

---

## 📊 Empirical Benchmarks (soc-pokec Dataset)

Tested on the **soc-pokec** power-law social graph (1.63M vertices, 30.6M edges, 1.78 Billion 2-hop motif paths):

| Metric | Native GraphFrames (Catalyst + AQE) | HyperGraphFrames (2D Hypercube) | Improvement |
| :--- | :--- | :--- | :--- |
| **Disk Spill** | **458.8 MB** | **0.0 MB** | **100% Elimination** |
| **Memory Spill** | **41.5 GB** | **0.0 MB** | **100% Elimination** |
| **Task Duration CV** | `0.1873` (High Variance) | `0.0210` (Uniform Tasks) | **8.9× Variance Reduction** |
| **End-to-End Latency** | `166.3 s` | **`84.2 s`** | **~2.0× Faster** |

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests
```

---

## 📄 License & Attribution

- **Author:** Rayirth Misra
- **License:** MIT License
