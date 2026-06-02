"""
drift-spark — Spark-native embedding lifecycle.

pip install drift-spark
import drift
"""

__version__ = "1.0.0"
__all__ = ["__version__", "DriftAdapter", "AdapterQualityError", "measure_arr"]

from .adapter import DriftAdapter
from .shadow_eval import AdapterQualityError, measure_arr
