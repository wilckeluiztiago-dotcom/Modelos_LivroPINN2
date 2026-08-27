# =============================================================================
# Módulo 06: Poços Inteligentes e Acoplamento Reservatório-Poço
# Autor: Luiz Tiago Wilcke
# Capítulo 6 do livro
# =============================================================================
"""Completações inteligentes, ICVs, ICDs, poços multissegmentados, DAE e acoplamento via PINNs."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from ..config.configuracoes import FISICA, PINN, GEOMETRIA
from ..utils.utilitarios import para_tensor, gradiente_autograd, LOGGER
from .modulo01_fundamentos import RedeBasePINN, FundamentosReservatorio

class PocoInteligente:
    """Modelagem de completação inteligente com ICVs e ICDs (Cap. 6)."""

    def __init__(self, n_segmentos: int = 5, n_icv: int = 3, n_icd: int = 8):
        self.n_segmentos = n_segmentos
        self.n_icv = n_icv
        self.n_icd = n_icd
        self.fundamentos = FundamentosReservatorio()
        self.cfg = FISICA
        self.geo = GEOMETRIA
        LOGGER.info(f"PocoInteligente ({n_segmentos} segs, {n_icv} ICVs, {n_icd} ICDs) - Luiz Tiago Wilcke")

    def queda_pressao_icd(self, vazao: float, a_icd: float = 0.5, b_icd: float = 1.5e-5) -> float:
        """Queda de pressão em dispositivo ICD (modelo de restrição). ΔP = a*Q + b*Q²"""
        return a_icd * vazao + b_icd * vazao**2

    def vazao_segmento_icv(self, pressao_reservatorio: float, pressao_tubing: float,
                           abertura_icv: float, ip_segmento: float) -> float:
        """Vazão controlada por ICV (0=fechado, 1=totalmente aberto)."""
        delta_p = pressao_reservatorio - pressao_tubing
        return max(0.0, ip_segmento * abertura_icv * delta_p)

    def modelo_multisegmentado(self, pressoes_res: np.ndarray, aberturas_icv: np.ndarray,
                               ips: np.ndarray, pressao_tubing: float) -> np.ndarray:
        """Calcula vazões por segmento em poço multissegmentado."""
        vazoes = np.zeros(self.n_segmentos)
        for i in range(self.n_segmentos):
            ab = aberturas_icv[i] if i < len(aberturas_icv) else 1.0
            vazoes[i] = self.vazao_segmento_icv(pressoes_res[i], pressao_tubing, ab, ips[i])
        return vazoes

    def residuo_dae_acoplamento(self, p_res: torch.Tensor, p_poco: torch.Tensor,
                                 q: torch.Tensor, ip: float) -> torch.Tensor:
        """Resíduo do sistema Algébrico-Diferencial (DAE) de acoplamento."""
        # Q = IP * (P_res - P_poco)  +  conservação
        return q - ip * (p_res - p_poco)

    def perda_carga_horizontal(self, vazao: float, comprimento: float, diametro: float,
                                densidade: float, viscosidade: float) -> float:
        """Perda de carga ao longo do trecho horizontal."""
        area = np.pi * (diametro/2)**2
        vel = vazao / max(area, 1e-8)
        re = densidade * vel * diametro / max(viscosidade, 1e-6)
        f = 0.316 / (re**0.25) if re > 2300 else 64/max(re,1)
        return f * (comprimento/diametro) * (densidade * vel**2 / 2)

    def resumo(self) -> Dict:
        return {
            "n_segmentos": self.n_segmentos,
            "n_icv": self.n_icv,
            "n_icd": self.n_icd,
            "tipo_completacao": self.geo.tipo_completacao,
            "autor": "Luiz Tiago Wilcke"
        }

class PINNPocoInteligente(RedeBasePINN):
    """PINN multiobjetivo para acoplamento reservatório-poço inteligente."""

    def __init__(self, n_saidas: int = 3):
        super().__init__(dim_entrada=3, dim_saida=n_saidas)  # (x,z,t) -> (P_res, P_poco, Q)
        self.fisica = PocoInteligente()

    def perda_fisica(self, x, z, t, ip=1e-12):
        entrada = torch.cat([x, z, t], dim=1)
        saida = self.forward(entrada)
        p_res, p_poco, q = saida[:,0:1], saida[:,1:2], saida[:,2:3]
        residuo = self.fisica.residuo_dae_acoplamento(p_res, p_poco, q, ip)
        return torch.mean(residuo**2)
