"""Run reports, inference preconditions and error handling."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from nlpoptnet import NLPOptNet
from nlpoptnet.api import reporting


def test_report_requires_a_completed_run(tmp_path):
    model = NLPOptNet()
    with pytest.raises(RuntimeError, match="No completed"):
        model.summary()
    with pytest.raises(RuntimeError, match="(?i)train or load"):
        model.predict([1])
    with pytest.raises(RuntimeError, match="(?i)train or load"):
        model.predict_with_info([1])
    with pytest.raises(ValueError, match="destination"):
        model.export_inference()
    model._metadata_path = str(tmp_path / "metadata.json")
    with pytest.raises(RuntimeError, match="history.csv not found"):
        model.plot_history(show=False)
    with pytest.raises(RuntimeError, match="summary.json not found"):
        model.summary()
    with pytest.raises(RuntimeError, match="(?i)train or load"):
        model.export_inference()


def test_report_uses_historical_violation_columns(tmp_path, capsys, monkeypatch):
    import matplotlib.pyplot as plt

    model = NLPOptNet()
    model._metadata_path = str(tmp_path / "metadata.json")
    (tmp_path / "summary.json").write_text(
        json.dumps({"model_name": "historical"}), encoding="utf-8"
    )
    model.summary()
    assert "historical" in capsys.readouterr().out
    assert reporting._fmt_time_sec(None) == "N/A"
    rows = [dict(epoch=1, train_objective=2, val_objective=3, train_eq_l2=0.25)]
    history_path = tmp_path / "history.csv"
    model._write_history_csv(history_path, [])
    assert not history_path.exists()
    model._write_history_csv(history_path, rows)
    np.testing.assert_allclose(
        model._history_column(model._read_history(), "train_eq_violation"), [0.25]
    )
    with pytest.raises(KeyError, match="missing"):
        model._history_column({}, "missing")
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fig, axes = model.plot_history(show=True, save_dir=tmp_path, bg="white")
    try:
        assert shown
        assert len(axes) == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fail", [False, True])
def test_inference_time_reports_include_errors_and_batch_size(
    tmp_path, monkeypatch, fail
):
    calls = []

    def predict(values, *, projection_backend):
        assert projection_backend == "jax"
        calls.append(np.asarray(values).shape)
        if fail:
            raise RuntimeError("backend unavailable")

    model = NLPOptNet()
    metadata = tmp_path / "metadata.json"
    result = model._estimate_inference_times(
        metadata_path=metadata, X_train=np.empty((0, 1)), batch_size=4
    )
    assert result["estimated_inference_samples"] == 0
    monkeypatch.setattr(
        reporting,
        "NLPOptNet",
        lambda: SimpleNamespace(
            load=lambda path, verbose: SimpleNamespace(predict=predict),
        ),
    )
    result = model._estimate_inference_times(
        metadata_path=metadata,
        X_train=np.array([[1.0], [2.0]]),
        batch_size=4,
        repetitions=2,
    )
    assert result["estimated_inference_samples"] == 2
    assert (4, 1) in calls
    for stage in ("single", "batch"):
        error = result[f"estimated_jax_{stage}_error"]
        timing = result[f"estimated_jax_{stage}_inference_time_sec"]
        if fail:
            assert error == "RuntimeError: backend unavailable"
            assert timing is None
        else:
            assert error is None and timing >= 0


def test_prediction_checks_parameter_shape():
    model = NLPOptNet()
    model._inference_state = SimpleNamespace(n_x=2)
    with pytest.raises(ValueError, match="2 parameter values"):
        model.predict([1])
    with pytest.raises(ValueError, match="2 parameter values"):
        model.predict_with_info(np.empty((0, 2)))
