"""Verify bounded benchmark storage with entirely synthetic arrays."""
import numpy as np

from .bars import file_digest
from .benchmark_matrix_synthetic import write_prediction


def test_repeated_prediction_uses_temporary_storage_and_matches_retained_sample(tmp_path):
    prediction = np.arange(24, dtype=np.float64).reshape(4, 6)
    origins = np.arange(4, dtype=np.int64)
    retained = tmp_path / "retained"
    scratch = tmp_path / "scratch"
    retained.mkdir()
    scratch.mkdir()
    first = write_prediction(prediction, origins, retained, retain=True)
    second = write_prediction(prediction, origins, scratch, retain=False)
    assert first["prediction_retained"] is True and second["prediction_retained"] is False
    assert second["prediction_path"] is None and list(scratch.iterdir()) == []
    assert first["synthetic_prediction_sha256"] == second["synthetic_prediction_sha256"]
    assert first["prediction_file_bytes"] == second["prediction_file_bytes"]
    assert file_digest(retained / "synthetic_prediction.npz") == first["synthetic_prediction_sha256"]
