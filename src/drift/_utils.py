"""Internal helpers shared across embed, watch, and migrate."""

from __future__ import annotations

import os

# macOS Homebrew Java 17 candidates — checked only when JAVA_HOME is unset.
# On Linux / Docker / CI, JAVA_HOME is typically set already; these are never used.
_JAVA17_CANDIDATES = [
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",  # macOS ARM
    "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",     # macOS Intel
]


def _maybe_set_java_home() -> None:
    """Set JAVA_HOME to a known Java 17 path only if it is not already set."""
    if os.environ.get("JAVA_HOME"):
        return
    for path in _JAVA17_CANDIDATES:
        if os.path.isdir(path):
            os.environ["JAVA_HOME"] = path
            return


def _get_spark(app_name: str = "drift"):
    """Return the active SparkSession or create a local one."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as err:
        raise ImportError(
            "pip install 'drift-spark[spark]' to use Spark-dependent features"
        ) from err

    spark = SparkSession.getActiveSession()
    if spark is None:
        _maybe_set_java_home()
        spark = (
            SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
    return spark
