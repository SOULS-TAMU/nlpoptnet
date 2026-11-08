"""Public NLPOptNet interfaces and prepared-projection resources."""

from .model import NLPOptNet, ProblemBuilder
from .objectives import nlpoptnet_loss
from purin.projections import LocalQuadratic, Sequential
from purin.solvers.diagonal_qp import QPData, QPStructure, PreparedDiagonalQP

__version__ = "0.2.0"
__all__ = [
    "NLPOptNet",
    "ProblemBuilder",
    "nlpoptnet_loss",
    "LocalQuadratic",
    "Sequential",
    "QPData",
    "QPStructure",
    "PreparedDiagonalQP",
]
