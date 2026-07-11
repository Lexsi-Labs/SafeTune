from setuptools import setup
import sys


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise RuntimeError("SafeTune requires Python 3.12 or higher")
    # Defer all metadata to pyproject.toml (PEP 621). This stub allows legacy
    # workflows like `pip install -e .` to continue working with PEP 517 builds.
    setup()
