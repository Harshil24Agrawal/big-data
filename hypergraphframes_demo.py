import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col
from graphframes import GraphFrame
from hypergraph import HyperGraphFrame

CSV_PATH = "/content/soc-pokec-edges.csv.gz"
MOTIF = "(a)-[e1]->(b); (b)-[e2]->(c)"

spark = (
    SparkSession.builder
    .appName("Pokec-HyperGraphFrames")
    .master("local[*]")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Use the same data preparation as the GraphFrames run.
edges = (
    spark.read.option("header", "true")
    .schema("src INT, dst INT")
    .csv(CSV_PATH)
    .repartition(200)
    .cache()
)
edges.count()

vertices = (
    edges.select(col("src").alias("id"))
    .union(edges.select(col("dst").alias("id")))
    .distinct()
    .cache()
)
vertices.count()

g = GraphFrame(broadcast(vertices), edges)

# Profile hubs using in-degree × out-degree, then apply 2D partitioning.
hg = HyperGraphFrame(
    g,
    work_threshold=1_000_000,
    p_cells=100,
    skip_id_only_vertex_join=True,
)

start = time.perf_counter()
path_count = hg.find(MOTIF).count()
elapsed_s = time.perf_counter() - start

print(f"Two-hop paths: {path_count:,}")
print(f"Elapsed time, including hub profiling: {elapsed_s:.2f} seconds")
print("\nSample of paths found by HyperGraphFrames (up to 10):")
# Displaying this sample runs an additional, bounded Spark action after the
# timed full-count action above; it is excluded from elapsed_s.
hg.show_motifs(MOTIF, n=10, truncate=False)
