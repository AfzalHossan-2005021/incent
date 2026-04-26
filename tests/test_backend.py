"""Tests for backend selection and the to_backend dispatch helper."""

from __future__ import annotations

import numpy as np
import ot
import pytest

from incent.utils import select_backend, to_backend


class TestSelectBackend:
    def test_returns_numpy_when_use_gpu_false(self) -> None:
        use_gpu, nx = select_backend(use_gpu=False, gpu_verbose=False)
        assert use_gpu is False
        assert nx is not None
        assert isinstance(nx, ot.backend.NumpyBackend)

    def test_returns_numpy_when_use_gpu_false_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Regression for the nx=None bug: prior to the fix, this code path
        # printed the "Tip:" line and then returned nx=None, crashing every
        # downstream to_backend(...) call.
        use_gpu, nx = select_backend(use_gpu=False, gpu_verbose=True)
        assert nx is not None
        captured = capsys.readouterr()
        # Either the CPU message or the CUDA tip — both are valid; nx must be non-None either way.
        assert "NumPy" in captured.out or "CUDA is available" in captured.out

    def test_falls_back_to_numpy_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        use_gpu, nx = select_backend(use_gpu=True, gpu_verbose=False)
        assert use_gpu is False
        assert isinstance(nx, ot.backend.NumpyBackend)


class TestToBackend:
    def test_numpy_roundtrip(self) -> None:
        nx = ot.backend.NumpyBackend()
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = to_backend(arr, nx)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, arr)

    def test_dtype_coercion(self) -> None:
        nx = ot.backend.NumpyBackend()
        arr = np.arange(4)
        out = to_backend(arr, nx, data_type=np.float32)
        assert out.dtype == np.float32

    def test_scipy_sparse_input(self) -> None:
        import scipy.sparse as sp

        nx = ot.backend.NumpyBackend()
        arr = sp.csr_matrix(np.eye(3, dtype=np.float32))
        out = to_backend(arr, nx, data_type=np.float32)
        np.testing.assert_array_equal(out, np.eye(3, dtype=np.float32))
