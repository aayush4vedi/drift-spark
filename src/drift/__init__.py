"""
drift-spark — Spark-native embedding lifecycle.

pip install drift-spark
import drift
"""

__version__ = "0.5.0"
__all__ = [
    "__version__",
    "embed", "watch", "migrate",
    "EmbedRun", "WatchRun", "MigrateRun",
    "DriftAdapter", "AdapterQualityError", "measure_arr",
    "Ledger",
]

from . import embed, migrate, watch
from .adapter import DriftAdapter
from .embed import EmbedRun
from .ledger import Ledger
from .migrate import MigrateRun
from .shadow_eval import AdapterQualityError, measure_arr
from .watch import WatchRun
