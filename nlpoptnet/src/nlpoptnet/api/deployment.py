"""Deployment resources for the NLPOptNet frontend."""

from __future__ import annotations
from nlpoptnet.serialization.artifacts import nlpoptnet_graph as nlpoptnet_graph


def export_inference(self, metadata_path=None):
    """Export backbone, scaling and all local QP layers as one JAX graph."""
    from ohagi.inference import export
    from nlpoptnet.serialization.artifacts import nlpoptnet_graph

    path = metadata_path or self._metadata_path
    if path is None:
        raise ValueError("Provide a destination metadata.json path.")
    return export(path, **nlpoptnet_graph(self))


def load_inference(metadata_path, *, device="cpu"):
    """Load the shared JAX predictor using only the model's metadata.json path."""
    from ohagi.inference import load

    return load(metadata_path, device=device)
