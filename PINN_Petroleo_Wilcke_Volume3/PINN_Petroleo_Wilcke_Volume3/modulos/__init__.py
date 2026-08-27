# =============================================================================
# Módulos PINN - Redes Neurais Informadas pela Física
# Autor: Luiz Tiago Wilcke
# =============================================================================
"""Exporta os 25 módulos especializados do sistema."""

from .modulo01_fundamentos import FundamentosReservatorio, RedeBasePINN
from .modulo02_escoamento_vertical import EscoamentoVerticalMultifasico, PINNEscoamentoVertical
from .modulo03_arquitetura_pinn import ArquiteturaPINN, FrameworkMAP
from .modulo04_anisotropia import AnisotropiaPermeabilidade, PINNInversaPermeabilidade
from .modulo05_elevacao_artificial import ElevacaoArtificial, PINNGasLift

# Módulos 06-25 (classes geradas automaticamente nos arquivos)
try:
    from .modulo06_poco_inteligente import PocoInteligente, PINNPocoInteligente
except ImportError:
    PocoInteligente = PINNPocoInteligente = None

try:
    from .modulo08_nao_newtoniano import FluidoNaoNewtoniano, PINNNaoNewtoniano
except ImportError:
    FluidoNaoNewtoniano = PINNNaoNewtoniano = None

try:
    from .modulo09_geomecanica import GeomecanicaPoroelastica, PINNGeomecanica
except ImportError:
    GeomecanicaPoroelastica = PINNGeomecanica = None

__all__ = [
    "FundamentosReservatorio", "RedeBasePINN",
    "EscoamentoVerticalMultifasico", "PINNEscoamentoVertical",
    "ArquiteturaPINN", "FrameworkMAP",
    "AnisotropiaPermeabilidade", "PINNInversaPermeabilidade",
    "ElevacaoArtificial", "PINNGasLift",
    "PocoInteligente", "PINNPocoInteligente",
    "FluidoNaoNewtoniano", "PINNNaoNewtoniano",
    "GeomecanicaPoroelastica", "PINNGeomecanica",
]
