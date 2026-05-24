# src/features/extractor_engine.py

from __future__ import annotations

import numpy as np

from src.features.registry import REGISTRY

# ---------------------------------------------------------------------------
# Constantes de inyección de envs
# ---------------------------------------------------------------------------
# Features que producen envolventes de banda (nombre → clave en envs dict).
_BAND_SOURCE_NAMES: dict[str, str] = {
    'alpha_envelope': 'alpha',
    'theta_envelope': 'theta',
    'beta_envelope':  'beta',
}

# Features que consumen el dict envs como entrada.
_DEPENDENT_FEATURE_NAMES: frozenset[str] = frozenset({
    'relative_power',
    'band_ratios',
    'hemispheric_asymmetry',
})


class ModularFeatureExtractor:
    """
    Motor modular de extracción de features.

    Reemplaza el extractor procedural legacy basado en if-statements.

    Env injection: las features dependientes de envolventes de banda
    (relative_power, band_ratios, hemispheric_asymmetry) reciben un dict
    `envs` pre-calculado una sola vez por ventana. Las envolventes base
    (alpha/theta/beta_envelope) no se recalculan si ya están en la lista
    de features activas.

    include_raw_envelopes (bool, default True):
        Si False, las envolventes absolutas (alpha/theta/beta_envelope) se
        calculan internamente para inyección en features dependientes pero
        NO se concatenan al vector de salida. Equivale a usar solo las
        features derivadas (relative_power, band_ratios, asymmetry).
        Configurable desde el YAML en el bloque `features:`.
    """

    def __init__(self, feature_config: dict):

        # Extraer flag antes de pasar al registry (key desconocida por registry)
        self._include_raw_envelopes: bool = bool(
            feature_config.get('include_raw_envelopes', True)
        )

        self.features = REGISTRY.resolve_from_yaml_flags(feature_config)

        if len(self.features) == 0:
            raise ValueError(
                "No se resolvió ninguna feature desde el YAML."
            )

        # Detectar si hay features dependientes de envs
        self._has_dependents = any(
            f.metadata.name in _DEPENDENT_FEATURE_NAMES for f in self.features
        )

        # Mapeo env_key → instancia de feature que la produce.
        # Reutiliza la instancia de self.features si ya está activa;
        # instancia una nueva (solo para envs, no para el vector) si no.
        self._band_env_sources: dict[str, object] = {}
        if self._has_dependents:
            active_by_name = {f.metadata.name: f for f in self.features}
            for feat_name, env_key in _BAND_SOURCE_NAMES.items():
                if feat_name in active_by_name:
                    self._band_env_sources[env_key] = active_by_name[feat_name]
                else:
                    self._band_env_sources[env_key] = REGISTRY.get(feat_name)

    def compute_epoch_features(
        self,
        epoch_data: np.ndarray,
        sfreq: float,
        win_samples: int,
        step_samples: int,
        edge_trim_windows: int = 0,
    ) -> np.ndarray:

        n_channels, n_samples = epoch_data.shape

        n_windows = (
            (n_samples - win_samples) // step_samples
        ) + 1

        all_windows = []

        for i in range(n_windows):

            start_idx = i * step_samples
            end_idx = start_idx + win_samples
            window_data = epoch_data[:, start_idx:end_idx]

            # ----------------------------------------------------------------
            # Pre-calcular envolventes de banda (una sola vez por ventana)
            # ----------------------------------------------------------------
            envs: dict[str, np.ndarray] = {}
            if self._has_dependents:
                for env_key, band_feat in self._band_env_sources.items():
                    env_result = band_feat.compute(
                        window_data=window_data, fs=sfreq
                    )
                    env_result = band_feat.validate_output(
                        env_result, n_channels=n_channels
                    )
                    envs[env_key] = env_result

            # ----------------------------------------------------------------
            # Calcular cada feature y concatenar al vector
            # ----------------------------------------------------------------
            feature_parts = []

            for feature in self.features:
                name = feature.metadata.name

                if name in _BAND_SOURCE_NAMES:
                    if self._has_dependents:
                        # Reutilizar resultado ya pre-calculado en envs
                        result = envs[_BAND_SOURCE_NAMES[name]]
                    else:
                        result = feature.compute(window_data=window_data, fs=sfreq)
                        result = feature.validate_output(result, n_channels=n_channels)
                    # Con include_raw_envelopes=False: calculadas pero excluidas del vector
                    if not self._include_raw_envelopes:
                        continue
                elif name in _DEPENDENT_FEATURE_NAMES:
                    result = feature.compute(
                        window_data=window_data, fs=sfreq, envs=envs
                    )
                    result = feature.validate_output(result, n_channels=n_channels)
                else:
                    result = feature.compute(window_data=window_data, fs=sfreq)
                    result = feature.validate_output(result, n_channels=n_channels)

                feature_parts.append(result)

            final_vector = np.concatenate(feature_parts)
            all_windows.append(final_vector)

        result = np.vstack(all_windows)

        if edge_trim_windows > 0:
            if 2 * edge_trim_windows >= result.shape[0]:
                raise ValueError(
                    f"edge_trim_windows={edge_trim_windows} eliminaría todas las "
                    f"ventanas ({result.shape[0]} disponibles). "
                    f"Reduce edge_trim o usa épocas más largas."
                )
            result = result[edge_trim_windows:-edge_trim_windows]

        return result

    def summary(self):

        print("\n=== MODULAR FEATURE EXTRACTOR ===")
        print(f"  include_raw_envelopes: {self._include_raw_envelopes}")

        for feature in self.features:

            meta = feature.metadata
            excluded = (
                meta.name in _BAND_SOURCE_NAMES and not self._include_raw_envelopes
            )
            tag = " [excluida del vector]" if excluded else ""

            print(
                f"- {meta.name:<25} "
                f"[{meta.category}] "
                f"HMM={meta.hmm_suitability}"
                f"{tag}"
            )

        print("=" * 40)
