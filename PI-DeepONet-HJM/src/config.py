"""
Módulo 01: Configurações e Hiperparâmetros
Autor: Luiz Tiago Wilcke
Projeto: PI-DeepONet para Dinâmica da Curva de Juros (HJM)
"""

from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class ConfiguracaoHJM:
    """Configuração central do modelo PI-DeepONet-HJM."""

    # Domínio temporal e de maturidade (em anos)
    t_min: float = 0.0
    t_max: float = 5.0
    T_min: float = 0.0
    T_max: float = 10.0

    # Sensores da curva forward inicial
    num_sensores: int = 50          # m sensores em f(0, ·)
    maturidades_sensores: List[float] = field(default_factory=list)

    # Arquitetura da DeepONet
    dim_latent: int = 64            # p = dimensão do produto interno
    camadas_branch: List[int] = field(default_factory=lambda: [128, 128, 128])
    camadas_trunk: List[int] = field(default_factory=lambda: [128, 128, 128])
    ativacao: str = "tanh"          # tanh, silu, gelu

    # Treinamento
    num_pontos_dominio: int = 4096
    num_pontos_contorno: int = 1024
    num_epocas_adam: int = 5000
    num_epocas_lbfgs: int = 500
    taxa_aprendizado: float = 1e-3
    peso_fisica: float = 1.0
    peso_dados: float = 10.0
    peso_livre_arbitragem: float = 5.0
    seed: int = 42

    # Volatilidade HJM (parametrização simples)
    sigma_constante: float = 0.01   # volatilidade constante para exemplo
    tipo_volatilidade: str = "constante"  # constante | exponencial | hull_white

    # Hardware
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if not self.maturidades_sensores:
            self.maturidades_sensores = list(
                torch.linspace(self.T_min + 1e-3, self.T_max, self.num_sensores).numpy()
            )


# Instância global padrão
CONFIG = ConfiguracaoHJM()
