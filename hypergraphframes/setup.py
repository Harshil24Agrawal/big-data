"""
setup.py
========
Backward-compatibility shim for editable installs and older build tools.
Build configuration is primarily defined in pyproject.toml (PEP 517/518/621).
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
