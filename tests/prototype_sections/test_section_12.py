import numpy as np
import pytest

from clean_core.config import ETA_L, FP32, POLE
from clean_core.geometry.lorentz import get_boost_frame, parallel_transport
from clean_core.prototypes.sections.section_12 import Section12Prototype


def _prototype():
    return Section12Prototype(
        input_dim=3, hidden_dim=2, rank=2, eta_m=0.2, eta_k=0.25, seed=17
    )


def _inputs():
    return {
        "error": np.linspace(-0.4, 0.5, ETA_L.size - 1, dtype=FP32),
        "activation": np.array([0.4, -0.2, 0.7], dtype=FP32),
        "hidden_activation": np.array([0.3, 0.8], dtype=FP32),
        "hidden_derivative": np.array([0.2, 0.4], dtype=FP32),
        "weights_1": np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.1]], dtype=FP32),
        "weights_2": np.linspace(
            -0.2, 0.3, (ETA_L.size - 1) * 2, dtype=FP32
        ).reshape(ETA_L.size - 1, 2),
        "mu_current": POLE.copy(),
        "mu_mean": POLE.copy(),
    }


def test_section_12_updates_oja_residual_and_both_kfac_factors():
    prototype = _prototype()
    prototype.dlr.V_basis = np.eye(ETA_L.size - 1, 2, dtype=FP32)
    prototype.dlr.s_values[:] = [0.3, 0.6]
    prototype.dlr.d_diag[:] = np.linspace(0.8, 1.2, ETA_L.size - 1)
    values = _inputs()
    error = values["error"]

    old_basis = prototype.dlr.V_basis.copy()
    old_s = prototype.dlr.s_values.copy()
    old_d = prototype.dlr.d_diag.copy()
    y = old_basis.T @ error
    raw_basis = old_basis + prototype.eta_m * (
        np.outer(error, y) - old_basis @ np.outer(y, y)
    )
    expected_basis, _ = np.linalg.qr(raw_basis, mode="reduced")
    expected_s = (1.0 - prototype.eta_m) * old_s + prototype.eta_m * (
        y**2 + 1e-5
    )
    expected_residual = error - expected_basis @ (expected_basis.T @ error)
    expected_d = (1.0 - prototype.eta_m) * old_d + prototype.eta_m * (
        expected_residual**2 + 1e-5
    )

    initial_a1 = prototype.kfac_layer1.A.copy()
    initial_b1 = prototype.kfac_layer1.B.copy()
    initial_a2 = prototype.kfac_layer2.A.copy()
    initial_b2 = prototype.kfac_layer2.B.copy()
    result = prototype.step(**values, precision_scalar=1.3)

    np.testing.assert_allclose(result["y"], y, atol=1e-6)
    np.testing.assert_allclose(result["basis"] @ result["basis"].T, expected_basis @ expected_basis.T, atol=1e-6)
    np.testing.assert_allclose(result["s_values"], expected_s, atol=1e-6)
    np.testing.assert_allclose(result["residual"], expected_residual, atol=1e-6)
    np.testing.assert_allclose(result["d_diag"], expected_d, atol=1e-6)
    np.testing.assert_allclose(
        result["basis"].T @ result["basis"], np.eye(prototype.rank), atol=1e-6
    )
    dlr_covariance = np.diag(result["d_diag"]) + result["basis"] @ np.diag(
        result["s_values"]
    ) @ result["basis"].T
    assert np.isfinite(dlr_covariance).all()
    assert np.linalg.eigvalsh(dlr_covariance).min() > 0.0
    np.testing.assert_allclose(result["precision"], result["precision"].T, atol=1e-6)
    assert np.linalg.eigvalsh(result["precision"]).min() > 0.0

    normalized_activation = values["activation"] / np.sqrt(
        1.0 + values["activation"] @ values["activation"]
    )
    normalized_g1 = result["g1"] / np.sqrt(1.0 + result["g1"] @ result["g1"])
    normalized_hidden = values["hidden_activation"] / np.sqrt(
        1.0 + values["hidden_activation"] @ values["hidden_activation"]
    )
    normalized_g2 = result["g2"] / np.sqrt(1.0 + result["g2"] @ result["g2"])
    np.testing.assert_allclose(
        result["A1"],
        (1.0 - prototype.eta_k) * initial_a1
        + prototype.eta_k * np.outer(normalized_activation, normalized_activation),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["B1"],
        (1.0 - prototype.eta_k) * initial_b1
        + prototype.eta_k * np.outer(normalized_g1, normalized_g1),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["A2"],
        (1.0 - prototype.eta_k) * initial_a2
        + prototype.eta_k * np.outer(normalized_hidden, normalized_hidden),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["B2"],
        (1.0 - prototype.eta_k) * initial_b2
        + prototype.eta_k * np.outer(normalized_g2, normalized_g2),
        atol=1e-6,
    )
    for factor in (result["A1"], result["B1"], result["A2"], result["B2"]):
        assert np.isfinite(factor).all()
        np.testing.assert_allclose(factor, factor.T, atol=1e-6)
        assert np.linalg.eigvalsh(factor).min() > 0.0
    assert not np.array_equal(result["A1"], initial_a1)
    assert not np.array_equal(result["B1"], initial_b1)
    assert not np.array_equal(result["A2"], initial_a2)
    assert not np.array_equal(result["B2"], initial_b2)


def test_section_12_transports_local_sensitivity_and_computes_natural_gradient():
    prototype = _prototype()
    values = _inputs()
    mu_current = POLE.copy()
    tangent = np.zeros_like(POLE)
    tangent[1] = 0.25
    mu_mean = np.cosh(0.25) * POLE + np.sinh(0.25) * tangent / 0.25
    values["mu_current"] = mu_current
    values["mu_mean"] = mu_mean
    precision = prototype.dlr.get_precision_matrix()
    g2_local = 1.3 * (precision @ values["error"])
    expected_g2 = get_boost_frame(mu_mean).T @ (
        ETA_L
        * parallel_transport(
            mu_current,
            mu_mean,
            get_boost_frame(mu_current) @ g2_local,
        )
    )

    result = prototype.step(**values, precision_scalar=1.3)
    np.testing.assert_allclose(result["g2_local"], g2_local, atol=1e-6)
    np.testing.assert_allclose(result["g2"], expected_g2, atol=1e-6)
    np.testing.assert_allclose(
        result["g1"],
        (values["weights_2"].T @ expected_g2) * values["hidden_derivative"],
        atol=1e-6,
    )

    natural_gradient = prototype.compute_natural_gradient(
        1, np.ones((2, 3), dtype=FP32)
    )
    assert natural_gradient.shape == (2, 3)
    assert np.isfinite(natural_gradient).all()
    assert np.linalg.norm(natural_gradient) <= 1.0 + 1e-6


def test_section_12_rejects_invalid_inputs_and_configuration():
    with pytest.raises(ValueError, match="eta_m"):
        Section12Prototype(3, 2, 2, eta_m=1.1)
    prototype = _prototype()
    values = _inputs()
    values["error"] = np.zeros(2, dtype=FP32)
    with pytest.raises(ValueError, match="error must have shape"):
        prototype.step(**values)
    with pytest.raises(ValueError, match="layer must be 1 or 2"):
        prototype.compute_natural_gradient(3, np.ones((2, 3)))
