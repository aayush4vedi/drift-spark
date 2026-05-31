import os
import shutil  # needed to delete the physical warehouse directory

# Must be set before PySpark starts the JVM gateway.
# PySpark 4.x requires Java 17 (class file 61.0).
_JAVA17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JAVA17):
    os.environ["JAVA_HOME"] = _JAVA17


from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("drift-local")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Wipe and recreate — safe to run repeatedly.
# Step 1: drop the metastore entry (CASCADE also drops tables inside).
spark.sql("DROP DATABASE IF EXISTS demo CASCADE")
# Step 2: delete the physical warehouse directory.
#   DROP DATABASE only removes the catalog entry — it leaves spark-warehouse/demo.db/
#   on disk. Spark then refuses CREATE DATABASE with LOCATION_ALREADY_EXISTS.
demo_dir = "spark-warehouse/demo.db"
if os.path.exists(demo_dir):
    shutil.rmtree(demo_dir)
# Step 3: recreate clean.
spark.sql("CREATE DATABASE demo")

# Create a tiny Delta-like table for demo (plain parquet — no delta required)
import pandas as pd
pdf = pd.DataFrame({"doc_id": [0,1,2], "body": ["reset password","billing issue","export CSV"]})
spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable("demo.support_docs")
print("Spark ready. Table 'demo.support_docs' created.")