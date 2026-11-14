"""Parameter sources and validation shared by training and scaling."""

import numpy as np
import pytest

from nlpoptnet import NLPOptNet


@pytest.mark.parametrize("format", ["ab", "matrix_rhs", "matrix_implicit", "array"])
def test_matrix_polytope_sampling(tmp_path, format):
    model = NLPOptNet(config={"seed": 7, "num_samples": 8})
    model.add_parameter("x")
    matrix = np.array([[1.0], [-1.0]])
    rhs = np.ones(2)
    if format in ("array", "matrix_implicit"):
        # Duplicating the zero RHS bounds pins x to zero.
        rhs = np.zeros(2)
    path = tmp_path / "polytope.npz"
    if format == "ab":
        np.savez(path, A=matrix, b=rhs)
    elif format == "matrix_rhs":
        np.savez(path, M=matrix, b=rhs)
    elif format == "matrix_implicit":
        np.savez(path, M=matrix)
    model.simplex(M=matrix if format == "array" else path).parameter_region("simplex")
    samples = model._load_or_sample_parameters(n_x=1)
    assert samples.shape == (8, 1)
    assert np.all(samples @ matrix.T <= rhs + 1e-10)


def test_box_from_extracted_constants_and_csv_switch(tmp_path):
    path = tmp_path / "problem.npz"
    np.savez(path, x_L=[1, 2], x_U=[2, 4], problem_type="qp")
    model = NLPOptNet(config={"num_samples": 8})
    model.add_parameter(["a", "b"])
    model.extract(path)
    assert model.problem_type == "qp"
    sample = model._load_or_sample_parameters(n_x=2)
    assert sample.shape == (8, 2)
    assert np.all((sample >= [1, 2]) & (sample <= [2, 4]))
    model.box(lower=model.x_L, upper=model.x_U, num_samples=8)
    np.testing.assert_array_equal(model._load_or_sample_parameters(n_x=2), sample)
    model.dataset(parameters=tmp_path / "samples.csv").parameter_region("data")
    model.parameter_region("box")
    np.testing.assert_array_equal(model._load_or_sample_parameters(n_x=2), sample)


def test_region_validation(tmp_path):
    model = NLPOptNet()
    with pytest.raises(ValueError, match="add_parameter"):
        model.box(lower=0, upper=1)
    p = model.add_parameter(["x", "y"])
    with pytest.raises(ValueError, match="needs lower"):
        model.box()
    with pytest.raises(ValueError, match="shape"):
        model.box(lower=[0, 0, 0], upper=1)
    for mode, message in [
        ("data", "dataset"),
        ("simplex", "simplex"),
        ("other", "one of"),
    ]:
        with pytest.raises(ValueError, match=message):
            model.parameter_region(mode)
    with pytest.raises(ValueError, match="Provide parameters"):
        model._load_or_sample_parameters(n_x=2)
    with pytest.raises(ValueError, match="either"):
        model.simplex(p.x <= 1, M=np.eye(2))
    with pytest.raises(ValueError, match="requires"):
        model.simplex()
    with pytest.raises(TypeError, match="expects parameter constraints"):
        model.simplex(lambda _: p.x <= 1)
    with pytest.raises(ValueError, match="2D"):
        model.simplex(M=[1, 2])
    missing = tmp_path / "missing.npy.npz"
    np.savez(missing, C=np.eye(2))
    with pytest.raises(ValueError, match="must contain"):
        model.simplex(M=missing)
    model.simplex(lambda: p.x == p.y, p.x >= 0, p.x <= 1, num_samples=8)
    samples = model._load_or_sample_parameters(n_x=2)
    np.testing.assert_allclose(samples[:, 0], samples[:, 1], atol=1e-10)
