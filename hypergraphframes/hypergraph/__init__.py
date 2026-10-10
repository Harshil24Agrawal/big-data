"""
HyperGraphFrames: Topology-Aware 2D Hypercube Partitioning for Skew-Resilient Graph Motif Finding.
"""

from hypergraph.core import HyperGraphFrame
from hypergraph.topology import profile_topology
from hypergraph.planner import run_hypercube_2hop, parse_2hop_motif

try:
    from importlib.metadata import version as _get_version, PackageNotFoundError
    try:
        __version__ = _get_version("hypergraphframes")
    except PackageNotFoundError:
        __version__ = "0.1.1"
except ImportError:
    __version__ = "0.1.1"
__author__ = "Rayirth Misra"
__all__ = [
    "HyperGraphFrame",
    "profile_topology",
    "run_hypercube_2hop",
    "parse_2hop_motif",
]
