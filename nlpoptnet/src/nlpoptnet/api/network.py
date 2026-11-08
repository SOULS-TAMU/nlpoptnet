"""NLPOptNet head dimensions supplied to OHAGI's shared Flax MLP."""

from ohagi.nn.adapters.projected_mlp import ProjectedMLP


def Backbone(*, p, n, me, mi, hidden_size, hidden_dim):
    """Construct OHAGI's Flax MLP with NLPOPTNet's primal and multiplier dimensions."""
    return ProjectedMLP(
        hidden_sizes=(hidden_size,) * hidden_dim,
        primal_size=n,
        equality_size=me,
        inequality_size=mi,
    )
