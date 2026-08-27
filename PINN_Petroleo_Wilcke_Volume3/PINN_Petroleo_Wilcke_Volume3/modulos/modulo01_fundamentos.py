# =============================================================================
# Módulo 01: Fundamentos da Física de Reservatórios e Mecânica dos Meios Porosos
# Autor: Luiz Tiago Wilcke
# Baseado no Capítulo 1 do livro
# =============================================================================
"""
Implementa:
- Propriedades petrofísicas (porosidade, permeabilidade)
- Lei de Darcy e Forchheimer
- Equação de difusividade monofásica
- Escoamento radial estacionário (Dupuit)
- Buckley-Leverett com histerese de Killough
- Equação de Estado Peng-Robinson (básico)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Callable
from ..config.configuracoes import FISICA, PINN
from ..utils.utilitarios import (
    para_tensor, gradiente_autograd, segunda_derivada,
    calcular_porosidade_pressao, grau_api_para_densidade, LOGGER
)

class FundamentosReservatorio:
    """Classe central de fundamentos físicos de reservatório."""

    def __init__(self, configuracao_fisica=None):
        self.cfg = configuracao_fisica or FISICA
        self.porosidade = self.cfg.porosidade_referencia
        self.permeabilidade = self.cfg.permeabilidade_horizontal
        self.viscosidade = self.cfg.viscosidade_oleo
        self.compressibilidade_total = (
            self.cfg.compressibilidade_fluido + self.cfg.compressibilidade_poros
        )
        LOGGER.info("FundamentosReservatorio inicializado por Luiz Tiago Wilcke")

    # ------------------------------------------------------------------
    # Propriedades Petrofísicas
    # ------------------------------------------------------------------
    def calcular_porosidade(self, pressao: float, pressao_ref: Optional[float] = None) -> float:
        """Eq. 1.6 - Porosidade compressível."""
        p0 = pressao_ref or self.cfg.pressao_inicial
        return calcular_porosidade_pressao(
            self.cfg.porosidade_referencia,
            self.cfg.compressibilidade_poros,
            pressao, p0
        )

    def tensor_permeabilidade(self, kx: Optional[float] = None, ky: Optional[float] = None,
                              kz: Optional[float] = None) -> np.ndarray:
        """Retorna tensor de permeabilidade anisotrópica (Eq. 1.7)."""
        kx = kx or self.cfg.permeabilidade_horizontal
        ky = ky or self.cfg.permeabilidade_horizontal
        kz = kz or self.cfg.permeabilidade_vertical
        return np.array([
            [kx, 0.0, 0.0],
            [0.0, ky, 0.0],
            [0.0, 0.0, kz]
        ])

    def densidade_oleo_api(self) -> float:
        """Densidade a partir do °API (Eq. 1.1)."""
        return grau_api_para_densidade(self.cfg.grau_api, self.cfg.densidade_agua)

    # ------------------------------------------------------------------
    # Lei de Darcy e Forchheimer
    # ------------------------------------------------------------------
    def velocidade_darcy(self, gradiente_p: np.ndarray, densidade: Optional[float] = None) -> np.ndarray:
        """Lei de Darcy vetorial (Eq. 1.8)."""
        rho = densidade or self.cfg.densidade_oleo
        g = np.array([0.0, 0.0, 9.81])
        K = self.tensor_permeabilidade()
        return - (K / self.viscosidade) @ (gradiente_p - rho * g)

    def velocidade_forchheimer(self, gradiente_p: float, beta_f: float = 1e8) -> float:
        """Equação de Forchheimer (Eq. 1.4) - 1D."""
        # Resolve equação quadrática: β ρ |u| u + (μ/k) u + dp/dx = 0
        a = beta_f * self.cfg.densidade_oleo
        b = self.viscosidade / self.permeabilidade
        c = gradiente_p
        # u = [-b + sqrt(b² - 4a c)] / (2a)  (raiz física)
        delta = b**2 - 4 * a * c
        if delta < 0:
            return 0.0
        u = (-b + np.sqrt(delta)) / (2 * a)
        return u

    # ------------------------------------------------------------------
    # Equação de Difusividade
    # ------------------------------------------------------------------
    def residuo_difusividade_1d(self, pressao: torch.Tensor, x: torch.Tensor,
                                t: torch.Tensor) -> torch.Tensor:
        """Resíduo da equação de difusividade monofásica 1D (Eq. 1.15)."""
        p_x = gradiente_autograd(pressao, x)
        p_xx = gradiente_autograd(p_x, x)
        p_t = gradiente_autograd(pressao, t)
        k = self.permeabilidade
        mu = self.viscosidade
        phi = self.porosidade
        ct = self.compressibilidade_total
        return (k / mu) * p_xx - phi * ct * p_t

    def residuo_difusividade_radial(self, pressao: torch.Tensor, r: torch.Tensor,
                                    t: torch.Tensor) -> torch.Tensor:
        """Resíduo em coordenadas radiais (Eq. 1.16 + temporal)."""
        p_r = gradiente_autograd(pressao, r)
        p_rr = gradiente_autograd(p_r, r)
        p_t = gradiente_autograd(pressao, t)
        k = self.permeabilidade
        mu = self.viscosidade
        phi = self.porosidade
        ct = self.compressibilidade_total
        # (1/r) d/dr (r dP/dr) = (φ ct μ / k) ∂P/∂t
        termo_espacial = p_rr + (1.0 / (r + 1e-8)) * p_r
        return (k / mu) * termo_espacial - phi * ct * p_t

    # ------------------------------------------------------------------
    # Solução Analítica Radial Estacionária (Dupuit)
    # ------------------------------------------------------------------
    def pressao_radial_estacionaria(self, r: np.ndarray, p_wf: float, p_e: float,
                                    r_w: Optional[float] = None, r_e: Optional[float] = None) -> np.ndarray:
        """Solução analítica logarítmica (Eq. 1.17)."""
        rw = r_w or self.cfg.raio_poco
        re = r_e or self.cfg.raio_drenagem
        return p_wf + (p_e - p_wf) * np.log(r / rw) / np.log(re / rw)

    def vazao_dupuit(self, p_e: float, p_wf: float, h: Optional[float] = None) -> float:
        """Equação de Dupuit (Eq. 1.18)."""
        h = h or self.cfg.espessura_reservatorio
        rw = self.cfg.raio_poco
        re = self.cfg.raio_drenagem
        k = self.permeabilidade
        mu = self.viscosidade
        return 2 * np.pi * k * h * (p_e - p_wf) / (mu * np.log(re / rw))

    # ------------------------------------------------------------------
    # Buckley-Leverett
    # ------------------------------------------------------------------
    def permeabilidade_relativa_corey(self, sw: np.ndarray, swc: float = 0.2,
                                      sor: float = 0.2, n_w: float = 2.0,
                                      n_o: float = 2.0, krw_max: float = 0.3,
                                      kro_max: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Curvas de permeabilidade relativa Corey."""
        s_star = (sw - swc) / (1 - swc - sor)
        s_star = np.clip(s_star, 0, 1)
        krw = krw_max * s_star ** n_w
        kro = kro_max * (1 - s_star) ** n_o
        return krw, kro

    def fracao_fluxo_agua(self, sw: np.ndarray, mu_w: Optional[float] = None,
                          mu_o: Optional[float] = None) -> np.ndarray:
        """Fração de fluxo fw (Eq. 1.20)."""
        mu_w = mu_w or self.cfg.viscosidade_agua
        mu_o = mu_o or self.cfg.viscosidade_oleo
        krw, kro = self.permeabilidade_relativa_corey(sw)
        return 1.0 / (1.0 + (kro * mu_w) / (krw * mu_o + 1e-12))

    def velocidade_frente_buckley(self, sw: np.ndarray, ut: float = 1e-5) -> np.ndarray:
        """Velocidade da frente dfw/dSw * ut / φ (Eq. 1.21)."""
        fw = self.fracao_fluxo_agua(sw)
        dfw = np.gradient(fw, sw)
        return (ut / self.porosidade) * dfw

    # ------------------------------------------------------------------
    # Histerese de Killough (simplificado)
    # ------------------------------------------------------------------
    def saturacao_gas_aprisionada_land(self, sg_hy: float, c_land: float = 2.0) -> float:
        """Isoterma de Land (Eq. 1.22)."""
        return sg_hy / (1.0 + c_land * sg_hy)

    def kr_gas_imbibicao_killough(self, sg: float, sg_hy: float, sg_t: float,
                                  kr_dr_max: float, alpha: float = 1.5) -> float:
        """Modelo de Killough (Eq. 1.24)."""
        if sg <= sg_t:
            return 0.0
        return kr_dr_max * ((sg - sg_t) / (sg_hy - sg_t + 1e-12)) ** alpha

    # ------------------------------------------------------------------
    # Peng-Robinson básico
    # ------------------------------------------------------------------
    def parametros_peng_robinson(self, t: float, tc: Optional[float] = None,
                                 pc: Optional[float] = None, omega: Optional[float] = None) -> Tuple[float, float]:
        """Parâmetros a(T) e b (Eq. 1.29-1.32)."""
        tc = tc or self.cfg.temperatura_critica
        pc = pc or self.cfg.pressao_critica
        omega = omega or self.cfg.fator_acentrico
        r = 8.314
        omega_a = 0.45724
        omega_b = 0.07780
        tr = t / tc
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + m * (1 - np.sqrt(tr)))**2
        a = omega_a * (r * tc)**2 / pc * alpha
        b = omega_b * r * tc / pc
        return a, b

    def pressao_peng_robinson(self, vm: float, t: float) -> float:
        """Pressão via EoS Peng-Robinson (Eq. 1.28)."""
        a, b = self.parametros_peng_robinson(t)
        r = 8.314
        return r * t / (vm - b) - a / (vm * (vm + b) + b * (vm - b))

    def resumo(self) -> Dict:
        """Retorna dicionário resumo dos fundamentos."""
        return {
            "porosidade": self.porosidade,
            "permeabilidade_m2": self.permeabilidade,
            "permeabilidade_mD": self.permeabilidade / 9.869e-16,
            "viscosidade_Pa_s": self.viscosidade,
            "compressibilidade_total": self.compressibilidade_total,
            "densidade_oleo": self.densidade_oleo_api(),
            "grau_api": self.cfg.grau_api,
            "autor": "Luiz Tiago Wilcke"
        }


class RedeBasePINN(nn.Module):
    """Rede neural base para PINNs com arquitetura MLP."""

    def __init__(self, dim_entrada: int = 2, dim_saida: int = 1,
                 camadas: int = 8, neuronios: int = 64, ativacao: str = "tanh"):
        super().__init__()
        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        camadas_lista = [nn.Linear(dim_entrada, neuronios)]
        for _ in range(camadas - 1):
            camadas_lista.append(nn.Linear(neuronios, neuronios))
        camadas_lista.append(nn.Linear(neuronios, dim_saida))
        self.rede = nn.ModuleList(camadas_lista)
        if ativacao == "tanh":
            self.ativacao = torch.tanh
        elif ativacao == "silu":
            self.ativacao = torch.nn.functional.silu
        else:
            self.ativacao = torch.tanh
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for camada in self.rede:
            if isinstance(camada, nn.Linear):
                nn.init.xavier_normal_(camada.weight)
                nn.init.zeros_(camada.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, camada in enumerate(self.rede[:-1]):
            x = self.ativacao(camada(x))
        return self.rede[-1](x)

    def predizer(self, coordenadas: np.ndarray) -> np.ndarray:
        """Predição em modo avaliação."""
        self.eval()
        with torch.no_grad():
            t = para_tensor(coordenadas, dispositivo=next(self.parameters()).device)
            return self.forward(t).cpu().numpy()
