import os
import sys

# 1. Map the Windows Hadoop utilities FIRST
def setup_env():
    if os.name == "nt":
        env_hadoop = os.environ.get("HADOOP_HOME")
        candidates = [
            *( [env_hadoop] if env_hadoop else [] ),
            os.path.join(os.getcwd(), "hadoop"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "hadoop")),
        ]
        hadoop_home = next((c for c in candidates if os.path.exists(c)), candidates[-1])
        bin_dir = os.path.join(hadoop_home, "bin")

        # Failsafe check
        if not os.path.exists(os.path.join(bin_dir, "winutils.exe")):
            print(f"🛑 STOP: Could not find winutils.exe at {bin_dir}\\winutils.exe")
            sys.exit(1)

        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        print("✅ Hadoop environment variables successfully mapped!")

setup_env()

# 2. NOW import Spark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PrepData").master("local[*]").getOrCreate()

# 3. Read and format the data
print("⏳ Reading dataset...")
df = spark.read.csv("soc-pokec-relationships.txt.gz", sep="\t")

edges = df.select(df["_c0"].cast("int").alias("src"), 
                  df["_c1"].cast("int").alias("dst"))

# 4. Save as Parquet with OVERWRITE
print("⏳ Writing Parquet file...")
edges.write.mode("overwrite").parquet("pokec_edges_parquet")
print("🎉 Successfully created pokec_edges_parquet!")
