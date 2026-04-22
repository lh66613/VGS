import numpy as np

from vgs.geometry import cumulative_explained_variance, effective_rank, projection_similarity


def test_effective_rank_concentrated_spectrum_is_low():
    rank = effective_rank(np.array([10.0, 0.1, 0.1]))
    assert rank < 1.2


def test_cumulative_explained_variance_ends_at_one():
    curve = cumulative_explained_variance(np.array([3.0, 4.0]))
    assert np.isclose(curve[-1], 1.0)


def test_projection_similarity_identical_bases():
    basis = np.eye(4)[:, :2]
    assert np.isclose(projection_similarity(basis, basis), 1.0)
