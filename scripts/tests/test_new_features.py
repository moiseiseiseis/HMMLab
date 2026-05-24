# scripts/tests/test_new_features.py
"""
Tests para los módulos nuevos: relative_power, band_ratios, hemispheric_asymmetry.

Ejecutar:
    pytest scripts/tests/test_new_features.py -v

O directamente:
    python scripts/tests/test_new_features.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import src.features  # dispara el registro automático

from src.features.registry import REGISTRY
from src.features.extractor_engine import ModularFeatureExtractor

# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

N_CH   = 19
N_SAMP = 300    # ~300ms a 1000Hz — ventana típica

RNG = np.random.default_rng(0)


def _make_window(n_ch: int = N_CH, n_samp: int = N_SAMP) -> np.ndarray:
    return RNG.standard_normal((n_ch, n_samp)).astype(np.float64) * 10e-6


def _make_envs(n_ch: int = N_CH, *, bands=('alpha', 'theta', 'beta')) -> dict:
    """Envolventes sintéticas (potencias positivas)."""
    return {
        b: np.abs(RNG.standard_normal(n_ch).astype(np.float64)) + 1e-6
        for b in bands
    }


# ---------------------------------------------------------------------------
# 1. relative_power: suma por canal ≈ 1.0
# ---------------------------------------------------------------------------

class TestRelativePower:

    def setup_method(self):
        self.feat = REGISTRY.get('relative_power')
        self.window = _make_window()
        self.envs   = _make_envs()

    def test_shape(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        assert result.shape == (N_CH * 3,), f"Esperado ({N_CH*3},), obtenido {result.shape}"

    def test_sum_per_channel_is_one(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        alpha_rel = result[:N_CH]
        theta_rel = result[N_CH:2*N_CH]
        beta_rel  = result[2*N_CH:]
        total = alpha_rel + theta_rel + beta_rel
        np.testing.assert_allclose(total, 1.0, atol=1e-8,
            err_msg="La suma alpha_rel+theta_rel+beta_rel debe ser 1.0 por canal")

    def test_all_finite(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        assert np.all(np.isfinite(result)), "relative_power devolvió NaN/Inf"

    def test_values_in_zero_one(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        assert np.all(result >= -1e-12) and np.all(result <= 1 + 1e-12), \
            "relative_power debe estar en [0, 1]"

    def test_missing_band_warns_and_returns_zeros(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.feat.compute(self.window, fs=256.0, envs={})
        assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
            "Debe emitir RuntimeWarning si faltan bandas"
        assert result.shape == (N_CH * 3,)


# ---------------------------------------------------------------------------
# 2. band_ratios: sin NaN/Inf, forma correcta
# ---------------------------------------------------------------------------

class TestBandRatios:

    def setup_method(self):
        self.feat = REGISTRY.get('band_ratios')
        self.window = _make_window()
        self.envs   = _make_envs()

    def test_shape(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        assert result.shape == (N_CH * 3,), f"Esperado ({N_CH*3},), obtenido {result.shape}"

    def test_no_nan_inf(self):
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        assert np.all(np.isfinite(result)), "band_ratios devolvió NaN/Inf"

    def test_log_identity(self):
        """log(θ/β) = log(θ/α) + log(α/β) — comprobación algebraica."""
        result = self.feat.compute(self.window, fs=256.0, envs=self.envs)
        log_ta = result[:N_CH]
        log_ab = result[N_CH:2*N_CH]
        log_tb = result[2*N_CH:]
        np.testing.assert_allclose(log_tb, log_ta + log_ab, atol=1e-10,
            err_msg="log(θ/β) debe ser igual a log(θ/α) + log(α/β)")

    def test_missing_band_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.feat.compute(self.window, fs=256.0, envs={'alpha': self.envs['alpha']})
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert result.shape == (N_CH * 3,)


# ---------------------------------------------------------------------------
# 3. hemispheric_asymmetry: en [-1, 1], forma dinámica
# ---------------------------------------------------------------------------

class TestHemisphericAsymmetry:

    def setup_method(self):
        self.feat = REGISTRY.get('hemispheric_asymmetry')
        self.window = _make_window()

    def test_shape_all_bands(self):
        envs = _make_envs()
        result = self.feat.compute(self.window, fs=256.0, envs=envs)
        assert result.shape == (7 * 3,), \
            f"Con 3 bandas esperado (21,), obtenido {result.shape}"

    def test_shape_one_band(self):
        envs = _make_envs(bands=('alpha',))
        result = self.feat.compute(self.window, fs=256.0, envs=envs)
        assert result.shape == (7,), \
            f"Con 1 banda esperado (7,), obtenido {result.shape}"

    def test_values_in_minus1_plus1(self):
        envs = _make_envs()
        result = self.feat.compute(self.window, fs=256.0, envs=envs)
        assert np.all(result >= -1 - 1e-9) and np.all(result <= 1 + 1e-9), \
            "Asimetría debe estar en (-1, 1)"

    def test_empty_with_no_bands_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.feat.compute(self.window, fs=256.0, envs={})
        assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
            "Debe emitir RuntimeWarning si no hay bandas activas"
        assert result.shape == (0,), "Sin bandas debe devolver array vacío"

    def test_symmetry_zero_when_lr_equal(self):
        """Si L==R, asimetría debe ser 0."""
        envs = {
            'alpha': np.ones(N_CH, dtype=np.float64),
        }
        result = self.feat.compute(self.window, fs=256.0, envs=envs)
        # Los 7 pares deben dar 0 (L==R==1)
        np.testing.assert_allclose(result, 0.0, atol=1e-8,
            err_msg="Asimetría debe ser 0 cuando L == R")


# ---------------------------------------------------------------------------
# 4. Shapes correctas con n_channels variable
# ---------------------------------------------------------------------------

class TestShapeConsistency:

    @pytest.mark.parametrize("n_ch", [19, 32, 64])
    def test_relative_power_shape_nchannel(self, n_ch):
        feat = REGISTRY.get('relative_power')
        window = _make_window(n_ch=n_ch)
        envs   = _make_envs(n_ch=n_ch)
        result = feat.compute(window, fs=256.0, envs=envs)
        assert result.shape == (n_ch * 3,)

    @pytest.mark.parametrize("n_ch", [19, 32, 64])
    def test_band_ratios_shape_nchannel(self, n_ch):
        feat = REGISTRY.get('band_ratios')
        window = _make_window(n_ch=n_ch)
        envs   = _make_envs(n_ch=n_ch)
        result = feat.compute(window, fs=256.0, envs=envs)
        assert result.shape == (n_ch * 3,)

    def test_asymmetry_always_7_per_band(self):
        feat = REGISTRY.get('hemispheric_asymmetry')
        window = _make_window()
        for n_bands, expected_len in [(1, 7), (2, 14), (3, 21)]:
            bands = ('alpha', 'theta', 'beta')[:n_bands]
            envs  = _make_envs(bands=bands)
            result = feat.compute(window, fs=256.0, envs=envs)
            assert result.shape == (expected_len,), \
                f"{n_bands} bandas → esperado ({expected_len},), obtenido {result.shape}"


# ---------------------------------------------------------------------------
# 5. Tolerancia a flags parciales (no crash si falta una banda)
# ---------------------------------------------------------------------------

class TestPartialFlagTolerance:

    def test_relative_power_partial_envs(self):
        feat = REGISTRY.get('relative_power')
        window = _make_window()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = feat.compute(window, fs=256.0, envs={'alpha': _make_envs()['alpha']})
        assert result.shape == (N_CH * 3,)
        assert np.all(np.isfinite(result))

    def test_band_ratios_partial_envs(self):
        feat = REGISTRY.get('band_ratios')
        window = _make_window()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = feat.compute(window, fs=256.0, envs={'theta': _make_envs()['theta']})
        assert result.shape == (N_CH * 3,)
        assert np.all(np.isfinite(result))

    def test_asymmetry_single_band_ok(self):
        feat = REGISTRY.get('hemispheric_asymmetry')
        window = _make_window()
        result = feat.compute(window, fs=256.0, envs={'beta': _make_envs()['beta']})
        assert result.shape == (7,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# 6. Test de integración: engine con 100 ventanas sintéticas
# ---------------------------------------------------------------------------

class TestEngineIntegration:

    N_WIN     = 100
    WIN_SAMP  = 300
    STEP_SAMP = 100
    SFREQ     = 1000.0

    def _make_epoch(self, n_ch: int = N_CH) -> np.ndarray:
        n_samples = self.WIN_SAMP + (self.N_WIN - 1) * self.STEP_SAMP
        return RNG.standard_normal((n_ch, n_samples)).astype(np.float64) * 10e-6

    def test_engine_all_new_features_with_envelopes(self):
        """Engine produce (100, D) con features nuevas + envelopes activos."""
        config = {
            'use_alpha':          True,
            'use_theta':          True,
            'use_beta':           True,
            'use_relative_power': True,
            'use_band_ratios':    True,
            'use_asymmetry':      True,
        }
        engine = ModularFeatureExtractor(config)
        epoch  = self._make_epoch()
        out    = engine.compute_epoch_features(
            epoch, self.SFREQ, self.WIN_SAMP, self.STEP_SAMP
        )

        assert out.shape[0] == self.N_WIN, \
            f"Esperado {self.N_WIN} ventanas, obtenido {out.shape[0]}"
        assert np.all(np.isfinite(out)), "Engine devolvió NaN/Inf"

        # Dimensión esperada: 3*N_CH (envs) + 3*N_CH (rel_pow) + 3*N_CH (ratios) + 21 (asym 3 bandas)
        expected_dim = 3 * N_CH + 3 * N_CH + 3 * N_CH + 7 * 3
        assert out.shape[1] == expected_dim, \
            f"Esperado dim={expected_dim}, obtenido {out.shape[1]}"

    def test_engine_new_features_without_envelope_flags(self):
        """Engine con features nuevas pero sin use_alpha/theta/beta en YAML.
        Las envolventes se computan internamente para inyección — no se añaden al vector."""
        config = {
            'use_relative_power': True,
            'use_band_ratios':    True,
        }
        engine = ModularFeatureExtractor(config)
        epoch  = self._make_epoch()
        out    = engine.compute_epoch_features(
            epoch, self.SFREQ, self.WIN_SAMP, self.STEP_SAMP
        )

        assert out.shape[0] == self.N_WIN
        assert np.all(np.isfinite(out))
        # Solo relative_power + band_ratios: 2 * 3 * N_CH
        expected_dim = 2 * 3 * N_CH
        assert out.shape[1] == expected_dim, \
            f"Esperado dim={expected_dim}, obtenido {out.shape[1]}"

    def test_engine_no_double_compute_with_shared_envelopes(self):
        """Activar alpha envelope + relative_power no duplica la dimensión de alpha."""
        config = {'use_alpha': True, 'use_relative_power': True}
        engine = ModularFeatureExtractor(config)
        epoch  = self._make_epoch()
        out    = engine.compute_epoch_features(
            epoch, self.SFREQ, self.WIN_SAMP, self.STEP_SAMP
        )
        # alpha_envelope (N_CH) + relative_power (3*N_CH) = 4*N_CH
        expected_dim = N_CH + 3 * N_CH
        assert out.shape[1] == expected_dim, \
            f"Esperado dim={expected_dim}, obtenido {out.shape[1]}"

    def test_engine_asymmetry_with_alpha_flag(self):
        """use_alpha + use_asymmetry: engine inyecta las 3 bandas → asym produce 7*3=21."""
        config = {'use_alpha': True, 'use_asymmetry': True}
        engine = ModularFeatureExtractor(config)
        epoch  = self._make_epoch()
        out    = engine.compute_epoch_features(
            epoch, self.SFREQ, self.WIN_SAMP, self.STEP_SAMP
        )
        # El engine pre-computa las 3 bandas para envs aunque solo use_alpha sea True.
        # Asimetría ve 3 bandas → 7*3 = 21; alpha_envelope → N_CH.
        expected_dim = N_CH + 7 * 3
        assert out.shape[1] == expected_dim, \
            f"Esperado dim={expected_dim}, obtenido {out.shape[1]}"

    def test_engine_edge_trim(self):
        """edge_trim_windows reduce el número de ventanas correctamente."""
        config = {'use_alpha': True, 'use_relative_power': True}
        engine = ModularFeatureExtractor(config)
        epoch  = self._make_epoch()
        trim   = 5
        out    = engine.compute_epoch_features(
            epoch, self.SFREQ, self.WIN_SAMP, self.STEP_SAMP,
            edge_trim_windows=trim,
        )
        assert out.shape[0] == self.N_WIN - 2 * trim


# ---------------------------------------------------------------------------
# Punto de entrada directo (sin pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Ejecutando tests directamente...\n")
    failed = 0

    test_classes = [
        TestRelativePower,
        TestBandRatios,
        TestHemisphericAsymmetry,
        TestShapeConsistency,
        TestPartialFlagTolerance,
        TestEngineIntegration,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]

        # Para TestShapeConsistency con parametrize, ejecutar manualmente
        for method_name in methods:
            m = getattr(obj, method_name)

            # Detectar métodos parametrizados por pytest y expandirlos manualmente
            if hasattr(m, "pytestmark"):
                print(f"  (skip directo — usar pytest para {method_name})")
                continue

            if hasattr(obj, 'setup_method'):
                obj.setup_method()
            try:
                m()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print("Todos los tests pasaron.")
    else:
        print(f"{failed} test(s) fallaron.")
    sys.exit(failed)
