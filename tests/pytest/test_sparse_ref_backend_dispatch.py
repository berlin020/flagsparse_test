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


def _spsv_coo_hipsparse_minimal_ready():
    backend, reason = spsv_mod._spsv_coo_sparse_ref_backend(
        torch.float32,
        torch.int32,
        op="non",
    )
    return backend == "hipsparse", reason


def _rand_like(dtype, shape, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    base = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, dtype=base, device=device)
    imag = torch.randn(shape, dtype=base, device=device)
    return torch.complex(real, imag)


def _build_triangular(n, dtype, device, lower):
    off = _rand_like(dtype, (n, n), device) * 0.02
    A = torch.tril(off) if lower else torch.triu(off)
    if torch.is_complex(A):
        diag = torch.rand(n, dtype=A.real.dtype, device=device) + 2.0
        A = A + torch.diag(torch.complex(diag, torch.zeros_like(diag)))
    else:
        diag = torch.rand(n, dtype=A.dtype, device=device) + 2.0
        A = A + torch.diag(diag)
    return A


def _apply_op(A, op_mode):
    if op_mode == "trans":
        return A.transpose(-2, -1)
    if op_mode == "conj":
        A_t = A.transpose(-2, -1)
        return A_t.conj() if torch.is_complex(A_t) else A_t
    return A


def _effective_upper(lower, op_mode):
    return lower if op_mode in ("trans", "conj") else not lower


def _rtol_atol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-4
    return 1e-10, 1e-10


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


def test_spsv_coo_sparse_ref_backend_selects_hipsparse_when_direct_supported(monkeypatch):
    monkeypatch.setattr(spsv_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsv_mod,
        "_hipsparse_spsv_skip_reason",
        lambda *args, **kwargs: None,
    )
    backend, reason = spsv_mod._spsv_coo_sparse_ref_backend(
        torch.float64,
        torch.int64,
        op="conj",
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


def test_spsv_coo_sparse_ref_backend_reports_reason_when_unsupported(monkeypatch):
    monkeypatch.setattr(spsv_mod, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        spsv_mod,
        "_hipsparse_spsv_skip_reason",
        lambda *args, **kwargs: "direct hipSPARSE COO SpSV unsupported",
    )
    backend, reason = spsv_mod._spsv_coo_sparse_ref_backend(
        torch.complex64,
        torch.int32,
        op="trans",
    )
    assert backend is None
    assert reason == "direct hipSPARSE COO SpSV unsupported"


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


@CUDA_REQUIRED
def test_spsv_csr_hipsparse_supported_combos():
    ready, reason = _spsv_hipsparse_minimal_ready()
    if not ready:
        pytest.skip(reason or "hipSPARSE CSR SpSV reference unavailable")

    device = torch.device("cuda")
    combos = (
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
        (torch.float64, torch.int32),
        (torch.float64, torch.int64),
        (torch.complex64, torch.int32),
        (torch.complex64, torch.int64),
        (torch.complex128, torch.int32),
        (torch.complex128, torch.int64),
    )
    for lower in (True, False):
        for op_mode in ("non", "trans", "conj"):
            for value_dtype, index_dtype in combos:
                backend, combo_reason = spsv_mod._spsv_csr_sparse_ref_backend(
                    value_dtype,
                    index_dtype,
                    index_dtype,
                    op=op_mode,
                )
                assert backend == "hipsparse", combo_reason
                A = _build_triangular(5, value_dtype, device, lower=lower)
                rhs = _rand_like(value_dtype, (5,), device)
                A_csr = A.to_sparse_csr()
                solution, meta = spsv_mod._spsv_csr_ref_hipsparse(
                    A_csr.values(),
                    A_csr.col_indices().to(index_dtype),
                    A_csr.crow_indices().to(index_dtype),
                    rhs,
                    (5, 5),
                    lower=lower,
                    unit_diagonal=False,
                    op=op_mode,
                    return_metadata=True,
                )
                expected = torch.linalg.solve_triangular(
                    _apply_op(A, op_mode),
                    rhs.unsqueeze(-1),
                    upper=_effective_upper(lower, op_mode),
                ).squeeze(-1)
                rtol, atol = _rtol_atol(value_dtype)
                assert meta["backend"] == "hipsparse"
                assert meta["format"] == "csr"
                assert torch.allclose(solution, expected, rtol=rtol, atol=atol)


@CUDA_REQUIRED
def test_spsv_coo_hipsparse_supported_combos():
    ready, reason = _spsv_coo_hipsparse_minimal_ready()
    if not ready:
        pytest.skip(reason or "hipSPARSE COO SpSV reference unavailable")

    device = torch.device("cuda")
    combos = (
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
        (torch.float64, torch.int32),
        (torch.float64, torch.int64),
        (torch.complex64, torch.int32),
        (torch.complex64, torch.int64),
        (torch.complex128, torch.int32),
        (torch.complex128, torch.int64),
    )
    for lower in (True, False):
        for op_mode in ("non", "trans", "conj"):
            for value_dtype, index_dtype in combos:
                backend, combo_reason = spsv_mod._spsv_coo_sparse_ref_backend(
                    value_dtype,
                    index_dtype,
                    op=op_mode,
                )
                assert backend == "hipsparse", combo_reason
                A = _build_triangular(5, value_dtype, device, lower=lower)
                rhs = _rand_like(value_dtype, (5,), device)
                A_coo = A.to_sparse_coo().coalesce()
                row, col = A_coo.indices()
                solution, meta = spsv_mod._spsv_coo_ref_hipsparse(
                    A_coo.values(),
                    row.to(index_dtype),
                    col.to(index_dtype),
                    rhs,
                    (5, 5),
                    lower=lower,
                    unit_diagonal=False,
                    op=op_mode,
                    return_metadata=True,
                )
                expected = torch.linalg.solve_triangular(
                    _apply_op(A, op_mode),
                    rhs.unsqueeze(-1),
                    upper=_effective_upper(lower, op_mode),
                ).squeeze(-1)
                rtol, atol = _rtol_atol(value_dtype)
                assert meta["backend"] == "hipsparse"
                assert meta["format"] == "coo"
                assert meta["canonical_format"] == "csr"
                assert torch.allclose(solution, expected, rtol=rtol, atol=atol)
