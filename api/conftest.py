"""Ensure the api/ directory is importable so tests can `import main`
regardless of the working directory pytest is launched from (e.g. in CI)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
