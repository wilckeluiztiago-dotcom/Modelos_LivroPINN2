# =============================================================================
# Módulo 02: Modelagem Dinâmica de Poços Verticais e Escoamento Vertical Multifásico
# Autor: Luiz Tiago Wilcke
# Capítulo 2 do livro
# =============================================================================
"""
Gradiente de pressão vertical (gravitacional + friccional + acelerativo)
Padrões de escoamento multifásico
Modelo de Dois Fluidos (Two-Fluid Model)
Balanço de energia térmica não-isotérmico
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from ..config.configuracoes import FISICA, PINN, GEOMETRIA
from ..utils.utilitarios import para_tensor, gradiente_autograd, LOGGER
from .modulo01_fundamentos import RedeBasePINN

class EscoamentoVerticalMultifasico:
    """Modelagem de escoamento vertical no poço (Cap. 2)."""

    def __init__(self):
        self.cfg = FISICA
        self.geo = GEOMETRIA
        self.g = 9.81
        LOGGER.info("EscoamentoVerticalMultifasico inicializado - Luiz Tiago Wilcke")

    def gradiente_gravitacional(self, densidade_mistura: float, inclinacao_rad: float = 0.0) -> float:
        """Componente gravitacional dp/dz (Eq. Cap. 2.2.1)."""
        return densidade_mistura * self.g * np.cos(inclinacao_rad)

    def fator_friccao_colebrook(self, reynolds: float, rugosidade_rel: float = 0.00015) -> float:
        """Fator de atrito de Colebrook-White (iterativo aproximado)."""
        if reynolds < 2300:
            return 64.0 / max(reynolds, 1.0)
        return (-1.8 * np.log10((rugosidade_rel / 3.7)**1.11 + 6.9 / reynolds))**(-2)

    def gradiente_friccional(self, densidade: float, velocidade: float, diametro: float,
                             viscosidade: float, rugosidade: float = 4.5e-5) -> float:
        """Componente friccional (Eq. Cap. 2.2.2)."""
        re = densidade * abs(velocidade) * diametro / max(viscosidade, 1e-6)
        f = self.fator_friccao_colebrook(re, rugosidade / diametro)
        return f * densidade * velocidade * abs(velocidade) / (2.0 * diametro)

    def gradiente_acelerativo(self, densidade: float, velocidade: float,
                              d_velocidade_dz: float) -> float:
        """Componente acelerativo (Eq. Cap. 2.2.3)."""
        return densidade * velocidade * d_velocidade_dz

    def gradiente_pressao_total(self, densidade: float, velocidade: float, diametro: float,
                                viscosidade: float, d_vel_dz: float = 0.0,
                                inclinacao_rad: float = 0.0) -> float:
        """Gradiente total de pressão vertical."""
        dp_grav = self.gradiente_gravitacional(densidade, inclinacao_rad)
        dp_fric = self.gradiente_friccional(densidade, velocidade, diametro, viscosidade)
        dp_acel = self.gradiente_acelerativo(densidade, velocidade, d_vel_dz)
        return dp_grav + dp_fric + dp_acel

    def densidade_mistura_holdup(self, holdup_liquido: float, dens_liq: float, dens_gas: float) -> float:
        """Densidade da mistura via holdup."""
        return holdup_liquido * dens_liq + (1 - holdup_liquido) * dens_gas

    def padrao_escoamento(self, vs_liq: float, vs_gas: float, diametro: float) -> str:
        """Classificação simplificada de padrões de fluxo."""
        if vs_gas < 0.1:
            return "bolhas"
        elif vs_gas < 1.0 and vs_liq > 0.3:
            return "slug"
        elif vs_gas > 5.0:
            return "anular"
        elif vs_liq < 0.05:
            return "neblina"
        else:
            return "intermitente"

    def equacoes_dois_fluidos_massa(self, alpha_g: torch.Tensor, rho_g: torch.Tensor,
                                    u_g: torch.Tensor, alpha_l: torch.Tensor,
                                    rho_l: torch.Tensor, u_l: torch.Tensor,
                                    z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resíduos de conservação de massa do Two-Fluid Model."""
        termo_g = alpha_g * rho_g
        termo_l = alpha_l * rho_l
        d_termo_g_dt = gradiente_autograd(termo_g, t)
        d_fluxo_g_dz = gradiente_autograd(termo_g * u_g, z)
        d_termo_l_dt = gradiente_autograd(termo_l, t)
        d_fluxo_l_dz = gradiente_autograd(termo_l * u_l, z)
        residuo_g = d_termo_g_dt + d_fluxo_g_dz
        residuo_l = d_termo_l_dt + d_fluxo_l_dz
        return residuo_g, residuo_l

    def balanco_energia_nao_isotermico(self, temperatura: torch.Tensor, z: torch.Tensor,
                                       t: torch.Tensor, velocidade: torch.Tensor,
                                       densidade: float, cp: float = 2000.0,
                                       k_termico: float = 0.15) -> torch.Tensor:
        """Resíduo do balanço de energia térmica (simplificado)."""
        t_t = gradiente_autograd(temperatura, t)
        t_z = gradiente_autograd(temperatura, z)
        t_zz = gradiente_autograd(t_z, z)
        return densidade * cp * (t_t + velocidade * t_z) - k_termico * t_zz


class PINNEscoamentoVertical(RedeBasePINN):
    """PINN especializada para gradiente de pressão vertical."""

    def __init__(self, **kwargs):
        super().__init__(dim_entrada=2, dim_saida=1, **kwargs)
        self.fisica = EscoamentoVerticalMultifasico()

    def perda_fisica(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Calcula perda do resíduo do gradiente de pressão."""
        entrada = torch.cat([z, t], dim=1)
        pressao = self.forward(entrada)
        p_z = gradiente_autograd(pressao, z)
        dens = self.fisica.cfg.densidade_oleo
        residuo = p_z + dens * 9.81
        return torch.mean(residuo**2)
