"""CUDA-graph shared-pool memory inspector (personal tooling).

Records, analyzes, and visualizes how SGLang's shared CUDA-graph memory pool is
used during graph capture, to surface inefficiency. See validator.py for the
feasibility-gate probe of the torch memory-snapshot schema.
"""
