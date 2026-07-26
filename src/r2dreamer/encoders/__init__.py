"""Flax encoder modules composed by the routed composite encoder.

Which branch consumes which observation key is declared by the adapters in
``src/adapters/`` and assembled in
``src/r2dreamer/encoders/routed_composite.py``; nothing is re-exported here so
that importing a single branch never drags in the others.
"""
