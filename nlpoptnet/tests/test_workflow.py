"""Exercise NLPOptNet's public training, projection and deployment contracts."""

from importlib.metadata import version

import jax
import numpy as np
import pytest

from nlpoptnet import NLPOptNet, __version__
from nlpoptnet.api.configuration import cfg_from_dict


jax.config.update("jax_enable_x64", True)


def make_model(**options):
    config = {
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "hidden_size": 8,
        "hidden_layers": 1,
        "cp_iters": 100,
        "device": "cpu",
        "verbose": False,
        **options,
    }
    return NLPOptNet(name="release_contract", config=config)


def define_qp(model):
    p = model.add_parameter(["p"])
    y = model.add_variable(["u", "v"])
    model.objective((y.u - p.p) ** 2 + 0.5 * y.v**2)
    model.constraints.equality.add(y.u + y.v == p.p)
    model.constraints.inequality.add(y.u <= 1.0)
    model.constraints.box.add(y.u >= 0, y.v >= 0, y.u <= 2, y.v <= 2)
    return model


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_training_and_metadata_roundtrip(tmp_path, monkeypatch, capsys, storage):
    monkeypatch.chdir(tmp_path)
    options = {"inequality_support": [[0]]} if storage == "sparse" else {}
    model = define_qp(make_model(**options, verbose=storage == "dense"))
    if storage == "dense":
        values = np.linspace(0.4, 1.4, 48)
        np.savetxt(tmp_path / "samples.csv", values, delimiter=",")
        model.dataset(parameters=tmp_path / "samples.csv")
    else:
        model.box(lower=0.4, upper=1.4, num_samples=48)
        model.scale(parameter=2, variable=2, objective=4, equality=2, inequality=2)
        model.build()
    result = model.optimize()
    assert len(result["history"]) == 1
    batch = np.array([[0.8], [0.6]])
    expected = np.asarray(model.predict(batch))
    assert expected.shape == (2, 2)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(axis=1), batch[:, 0], atol=1e-5)
    model.summary()
    assert "Training Summary" in capsys.readouterr().out
    model.plot_history(show=False)
    restored = NLPOptNet().load(result["metadata_path"], verbose=storage == "dense")
    np.testing.assert_allclose(restored.predict(batch), expected, rtol=1e-8, atol=1e-9)
    info = restored.predict_with_info(batch)
    np.testing.assert_allclose(info["prediction"], expected, rtol=1e-8, atol=1e-9)
    assert np.all(info["layers_used"] == restored._inference_state.cfg.k_layer)
    assert info["cp_iterations"].shape == (2, restored._inference_state.cfg.k_layer)
    assert np.all(info["cp_iterations"] <= 100)
    adaptive = restored.predict_with_info(batch[0], adaptive=True, max_layers=4)
    assert adaptive["primal_feasible"]
    assert adaptive["layers_used"] == 1
    assert adaptive["cp_iterations"].shape == (1,)
    assert adaptive["prediction"].shape == (2,)
    with pytest.raises(ValueError, match="max_layers"):
        restored.predict_with_info(batch, adaptive=True, max_layers=0)
    with pytest.raises(ValueError, match="max_layers"):
        restored.predict_with_info(batch, max_layers=2)
    with pytest.raises(ValueError, match="tolerances"):
        restored.predict_with_info(batch, stationarity_tolerance=-1.0)
    metadata = tmp_path / "deployment" / "metadata.json"
    model.export_inference(metadata)
    predictor = model.load_inference(metadata)
    for sample in batch:
        # Compare the same batch shape because finite CP norm estimates may
        # depend on batching.
        answer = np.asarray(model.predict(sample[None, :]))[0]
        np.testing.assert_allclose(predictor(sample), answer, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    ("raw", "mode", "fixed"),
    [
        ({}, "fixed", True),
        ({"IS_FIXED": False}, "accelerated", False),
        ({"cp_mode": " FIXED ", "IS_FIXED": False}, "fixed", True),
    ],
)
def test_projection_mode_aliases(raw, mode, fixed):
    cfg = cfg_from_dict(raw)
    assert cfg.cp_mode == mode
    assert cfg.IS_FIXED is fixed


def test_sparse_support_is_normalized():
    cfg = cfg_from_dict({"inequality_support": [[0, 3], []]})
    assert cfg.inequality_support == ((0, 3), ())


@pytest.mark.parametrize(
    ("config", "message"),
    [({"cp_mode": "other"}, "cp_mode"), ({"k_layer": 0}, "k_layer")],
)
def test_rejects_invalid_projection_options(config, message):
    with pytest.raises(ValueError, match=message):
        cfg_from_dict(config)


def test_distribution_version_matches_import():
    assert __version__ == version("nlpoptnet")
