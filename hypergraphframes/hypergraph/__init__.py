"""
HyperGraphFrames: Topology-Aware 2D Hypercube Partitioning for Skew-Resilient Graph Motif Finding.
"""

from hypergraph.core import HyperGraphFrame
from hypergraph.topology import profile_topology
from hypergraph.planner import run_hypercube_2hop, parse_2hop_motif

def _resolve_version() -> str:
    try:
        from importlib.metadata import version as _get_version
        return _get_version("hypergraphframes")
    except Exception:
        import os
        import re
        pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
        if os.path.exists(pyproject_path):
            try:
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    match = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
                    if match:
                        return match.group(1)
            except Exception:
                pass
        return "0.1.2"


__version__ = _resolve_version()
__author__ = "Rayirth Misra"
__all__ = [
    "HyperGraphFrame",
    "profile_topology",
    "run_hypercube_2hop",
    "parse_2hop_motif",
]
