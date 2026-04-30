"""Parity training shim — JAX R2 vs PT R2 on Crafter."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.parity.train_parity import run

if __name__ == "__main__":
    run(argv=None)
