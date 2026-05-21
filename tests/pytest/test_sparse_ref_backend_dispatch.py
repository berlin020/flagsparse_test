import pytest
import torch

import flagsparse.sparse_operations.spsm as spsm_mod
import flagsparse.sparse_operations.spsv as spsv_mod

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _spsv_hipsparse_minimal_ready():
    backend, reason = spsv_mod._spsv_csr_sparse_ref_backend(
        torch.float32,
        torch.int32,
        torch.int32,
        op="non",
    )
    return backend == "hipsparse", reason


def test_spsv_csr_sparse_ref_backend_selects_hipsparse_when_direct_supported(monkeypatch):
    monkeypatch.setattr(spsv_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsv_mod,
        "_hipsparse_spsv_skip_reason",
        lambda *args, **kwargs: None,
    )
    backend, reason = spsv_mod._spsv_csr_sparse_ref_backend(
        torch.float32,
        torch.int32,
        torch.int32,
        op="non",
    )
    assert backend == "hipsparse"
    assert reason is None


def test_spsv_csr_sparse_ref_backend_reports_reason_when_unsupported(monkeypatch):
    monkeypatch.setattr(spsv_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsv_mod,
        "_hipsparse_spsv_skip_reason",
        lambda *args, **kwargs: "direct hipSPARSE CSR SpSV unsupported",
    )
    backend, reason = spsv_mod._spsv_csr_sparse_ref_backend(
        torch.complex128,
        torch.int64,
        torch.int64,
        op="conj",
    )
    assert backend is None
    assert reason == "direct hipSPARSE CSR SpSV unsupported"


def test_spsm_csr_sparse_ref_backend_selects_hipsparse_when_direct_supported(monkeypatch):
    monkeypatch.setattr(spsm_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsm_mod,
        "_hipsparse_spsm_skip_reason",
        lambda *args, **kwargs: None,
    )
    backend, reason = spsm_mod._spsm_csr_sparse_ref_backend(
        torch.float64,
        torch.int32,
        torch.int32,
    )
    assert backend == "hipsparse"
    assert reason is None


def test_spsm_csr_sparse_ref_backend_reports_reason_when_unsupported(monkeypatch):
    monkeypatch.setattr(spsm_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsm_mod,
        "_hipsparse_spsm_skip_reason",
        lambda *args, **kwargs: "direct hipSPARSE CSR SpSM unsupported",
    )
    backend, reason = spsm_mod._spsm_csr_sparse_ref_backend(
        torch.float32,
        torch.int32,
        torch.int32,
    )
    assert backend is None
    assert reason == "direct hipSPARSE CSR SpSM unsupported"


@CUDA_REQUIRED
def test_spsv_csr_hipsparse_minimal_reference_sample():
    ready, reason = _spsv_hipsparse_minimal_ready()
    if not ready:
        pytest.skip(reason or "hipSPARSE CSR SpSV reference unavailable")

    device = torch.device("cuda")
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    rhs = torch.tensor([14.0, 15.0], dtype=torch.float32, device=device)

    solution, meta = spsv_mod._spsv_csr_ref_hipsparse(
        data,
        indices,
        indptr,
        rhs,
        (2, 2),
        lower=True,
        unit_diagonal=False,
        op="non",
        return_metadata=True,
    )
    expected = torch.tensor([4.0, 5.0], dtype=torch.float32, device=device)
    assert meta["backend"] == "hipsparse"
    assert torch.allclose(solution, expected, rtol=1e-6, atol=1e-6)
