import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col
from graphframes import GraphFrame

CSV_PATH = "/content/soc-pokec-edges.csv.gz"
MOTIF = "(a)-[e1]->(b); (b)-[e2]->(c)"

spark = (
    SparkSession.builder
    .appName("Pokec-Native-GraphFrames")
    .master("local[*]")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Load and materialize the same edge and vertex data before timing.
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

start = time.perf_counter()
motifs = g.find(MOTIF)
path_count = motifs.count()
elapsed_s = time.perf_counter() - start

print(f"Two-hop paths: {path_count:,}")
print(f"Elapsed time: {elapsed_s:.2f} seconds")
print("\nSample of paths found by native GraphFrames (up to 10):")
motifs.show(10, truncate=False)
