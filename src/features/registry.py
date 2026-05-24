# src/features/registry.py
"""
Registro central de features del pipeline EEG-HMM.

Responsabilidades:
  1. Mantener un dict {nombre -> clase} de features registradas.
  2. Resolver el bloque `features:` de un YAML (flags booleanos O lista explicita)
     en instancias listas para compute().
  3. Exponer metadata de todas las features registradas.

NO computa nada.
NO sabe de MNE ni de numpy mas alla de type hints.
NO tiene dependencias externas mas alla de base.py.

REGISTRO AUTOMATICO:
  Cada modulo de feature usa el decorador @REGISTRY.register sobre su clase.
  Para que el registro se ejecute, los modulos deben importarse.
  Esto ocurre en src/features/__init__.py, que importa todos los submodulos.
  El usuario NUNCA necesita importar los modulos de feature individualmente.

COMPATIBILIDAD HACIA ATRAS:
  Los YAMLs actuales usan flags booleanos:
      features:
        use_hjorth: true
        use_alpha:  false
  Estos siguen funcionando gracias a YAML_FLAG_TO_FEATURES.

SINTAXIS NUEVA (opcional):
  Los YAMLs nuevos pueden usar lista explicita:
      features:
        active:
          - hjorth_mobility
          - hjorth_complexity
          - alpha_envelope
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.features.base import BaseFeature, FeatureMetadata


# ---------------------------------------------------------------------------
# Mapeo de compatibilidad: flag YAML -> lista de nombres canonicos
# ---------------------------------------------------------------------------
# Este mapeo es el UNICO lugar donde se define la equivalencia entre
# los flags booleanos del YAML actual y los nombres canonicos del registry.
#
# Regla: si un flag activa multiples features (ej: use_hjorth activa mobility
# Y complexity), todas van aqui en la misma lista.
#
# Para agregar un nuevo flag al YAML antiguo, solo agrega una entrada aqui.
# NO hay que modificar nada mas.

YAML_FLAG_TO_FEATURES: dict[str, list[str]] = {
    "use_alpha":   ["alpha_envelope"],
    "use_theta":   ["theta_envelope"],
    "use_beta":    ["beta_envelope"],
    "use_hjorth":  ["hjorth_mobility", "hjorth_complexity"],
    "use_entropy":        ["temporal_entropy"],
    "use_relative_power": ["relative_power"],
    "use_band_ratios":    ["band_ratios"],
    "use_asymmetry":      ["hemispheric_asymmetry"],
    # Futuros flags (agregar aqui cuando se implementen los modulos):
    # "use_gfp":    ["gfp"],
    # "use_rms":    ["rms"],
    # "use_spectral_entropy": ["spectral_entropy"],
}


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    Registro central de features disponibles en el pipeline.

    Uso como decorador (en cada modulo de feature):

        from src.features.registry import REGISTRY

        @REGISTRY.register
        class HjorthMobilityFeature(BaseFeature):
            metadata = FeatureMetadata(name="hjorth_mobility", ...)
            def compute(...): ...

    Uso para resolver features desde YAML:

        features = REGISTRY.resolve_from_yaml_flags(exp_cfg['features'])
        # -> [HjorthMobilityFeature(), HjorthComplexityFeature()]

    Uso para inspeccion:

        REGISTRY.print_summary()
        all_meta = REGISTRY.all_metadata()
    """

    def __init__(self):
        # Dict interno: nombre_canonico -> clase (NO instancia)
        # Se instancia bajo demanda en resolve() para permitir parametros futuros.
        self._registry: dict[str, type[BaseFeature]] = {}

    # ------------------------------------------------------------------
    # Registro
    # ------------------------------------------------------------------

    def register(self, feature_cls: type) -> type:
        """
        Registra una clase de feature. Usable como decorador o llamada directa.

        Args:
            feature_cls: Subclase de BaseFeature con atributo `metadata` definido.

        Returns:
            La misma clase (para que funcione como decorador sin side effects).

        Raises:
            AttributeError: Si la clase no tiene atributo `metadata`.
            AttributeError: Si metadata no tiene atributo `name`.
            ValueError:     Si el nombre ya esta registrado (evita colisiones silenciosas).
        """
        if not hasattr(feature_cls, "metadata"):
            raise AttributeError(
                f"La clase '{feature_cls.__name__}' no tiene atributo 'metadata'. "
                f"Toda feature debe definir metadata = FeatureMetadata(...) "
                f"como atributo de clase."
            )

        name = feature_cls.metadata.name

        if not name:
            raise ValueError(
                f"La clase '{feature_cls.__name__}' tiene metadata.name vacio."
            )

        if name in self._registry:
            existing_cls = self._registry[name]
            raise ValueError(
                f"Nombre de feature '{name}' ya registrado por '{existing_cls.__name__}'. "
                f"Intento de re-registro por '{feature_cls.__name__}'. "
                f"Cada feature debe tener un nombre canonico unico."
            )

        self._registry[name] = feature_cls
        return feature_cls

    # ------------------------------------------------------------------
    # Resolucion desde YAML
    # ------------------------------------------------------------------

    def resolve_from_yaml_flags(self, flags: dict) -> list[BaseFeature]:
        """
        Convierte el bloque `features:` del YAML en instancias de BaseFeature.

        Soporta DOS sintaxis:

        SINTAXIS 1 -- Flags booleanos (YAML actual, compatibilidad total):
            features:
              use_hjorth: true
              use_alpha:  false
              use_entropy: true

        SINTAXIS 2 -- Lista explicita (YAMLs nuevos):
            features:
              active:
                - hjorth_mobility
                - hjorth_complexity

        La deteccion es automatica: si el dict tiene clave 'active', usa sintaxis 2.
        En caso contrario, asume sintaxis 1.

        Args:
            flags: Dict correspondiente a la seccion `features:` del YAML.
                   Puede ser None o vacio -> devuelve lista vacia.

        Returns:
            Lista de instancias BaseFeature, en el orden en que se resolvieron.
            El orden es determinista: sigue el orden de YAML_FLAG_TO_FEATURES.

        Raises:
            KeyError: Si un nombre en la lista 'active' no esta registrado.
            RuntimeError: Si flags activos pero no hay features registradas
                          (indica que los modulos no se importaron).
        """
        if not flags:
            return []

        # Detectar sintaxis
        if "active" in flags:
            return self._resolve_explicit_list(flags["active"])
        else:
            return self._resolve_boolean_flags(flags)

    def _resolve_boolean_flags(self, flags: dict) -> list[BaseFeature]:
        """Resuelve sintaxis 1: flags booleanos del YAML actual."""
        resolved_names: list[str] = []

        for flag_name, feature_names in YAML_FLAG_TO_FEATURES.items():
            if flags.get(flag_name, False):
                resolved_names.extend(feature_names)

        # Eliminar duplicados manteniendo orden (por si algun flag futuro solapa)
        seen = set()
        unique_names = []
        for n in resolved_names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        return self._instantiate_names(unique_names, context="flags booleanos del YAML")

    def _resolve_explicit_list(self, active_list: list[str]) -> list[BaseFeature]:
        """Resuelve sintaxis 2: lista explicita de nombres canonicos."""
        return self._instantiate_names(active_list, context="lista 'active' del YAML")

    def _instantiate_names(self, names: list[str], context: str) -> list[BaseFeature]:
        """
        Instancia clases por nombre. Centraliza el manejo de errores.

        Args:
            names:   Lista de nombres canonicos a resolver.
            context: String descriptivo para mensajes de error (quien llama).
        """
        if not self._registry:
            raise RuntimeError(
                "El registry esta vacio. "
                "Asegurate de importar src.features antes de resolver features. "
                "El __init__.py de src/features/ debe importar todos los submodulos."
            )

        instances = []
        for name in names:
            if name not in self._registry:
                available = sorted(self._registry.keys())
                raise KeyError(
                    f"Feature '{name}' (desde {context}) no encontrada en el registry.\n"
                    f"Features registradas: {available}\n"
                    f"Olvidaste agregar el modulo al __init__.py de src/features/?"
                )
            instances.append(self._registry[name]())

        return instances

    # ------------------------------------------------------------------
    # Acceso y consulta
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseFeature:
        """
        Devuelve una nueva instancia de la feature por nombre canonico.

        Args:
            name: Nombre canonico (ej: "hjorth_mobility").

        Returns:
            Nueva instancia de la subclase de BaseFeature correspondiente.

        Raises:
            KeyError: Si el nombre no esta registrado.
        """
        if name not in self._registry:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"Feature '{name}' no registrada. Disponibles: {available}"
            )
        return self._registry[name]()

    def all_metadata(self) -> list[FeatureMetadata]:
        """
        Devuelve la metadata de todas las features registradas,
        ordenada alfabeticamente por nombre canonico.

        Util para:
            - Generar reportes de diagnostico
            - Documentacion automatica
            - Comparar metadata declarada vs metricas empiricas
        """
        return [
            cls.metadata
            for cls in sorted(self._registry.values(), key=lambda c: c.metadata.name)
        ]

    def registered_names(self) -> list[str]:
        """Lista de nombres canonicos registrados, ordenada alfabeticamente."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """True si el nombre canonico esta registrado."""
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def print_summary(self) -> None:
        """
        Imprime un resumen de todas las features registradas.
        Util para debugging y verificacion post-instalacion.
        """
        print(f"\n{'='*60}")
        print(f"  Feature Registry -- {len(self._registry)} features registradas")
        print(f"{'='*60}")

        if not self._registry:
            print("  (vacio -- importaste src.features?)")
            print(f"{'='*60}\n")
            return

        # Agrupar por categoria
        by_category: dict[str, list] = {}
        for cls in sorted(self._registry.values(), key=lambda c: c.metadata.name):
            cat = cls.metadata.category
            by_category.setdefault(cat, []).append(cls.metadata)

        for category, metas in sorted(by_category.items()):
            print(f"\n  [{category.upper()}]")
            for meta in metas:
                suitability_icon = {
                    "high":   "OK",
                    "medium": "WARN",
                    "low":    "ERROR",
                }.get(meta.hmm_suitability, "?")
                scaling_tag = f"[{meta.recommended_scaling}]"
                print(
                    f"    {suitability_icon} {meta.name:<30} "
                    f"{scaling_tag:<15} "
                    f"{'n_ch' if meta.n_channels_dependent else 'global'}"
                )

        print(f"\n  Leyenda: OK alta suitability HMM | WARN  media | ERROR baja")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Instancia global (singleton de modulo)
# ---------------------------------------------------------------------------
# Python garantiza que este modulo se importa una sola vez por proceso.
# No se necesita implementacion Singleton formal.
# Todo el codigo importa REGISTRY desde aqui.

REGISTRY = FeatureRegistry()
