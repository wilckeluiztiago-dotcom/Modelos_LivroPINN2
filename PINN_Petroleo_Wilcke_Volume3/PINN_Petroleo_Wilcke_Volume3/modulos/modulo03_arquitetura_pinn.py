# =============================================================================
# Módulo 03: Formulação de Redes Neurais Informadas pela Física (PINNs)
# Autor: Luiz Tiago Wilcke
# Capítulo 3 do livro
# =============================================================================
"""Arquitetura PINN, função de perda multiobjetivo, treinamento, MAP e viés-variância."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from ..config.configuracoes import PINN, FISICA
from ..utils.utilitarios import para_tensor, gradiente_autograd, LOGGER, salvar_checkpoint
from .modulo01_fundamentos import RedeBasePINN

class ArquiteturaPINN(nn.Module):
    """Arquitetura completa de PINN com perda multiobjetivo (Cap. 3)."""

    def __init__(self, dim_entrada: int = 2, dim_saida: int = 1,
                 camadas: int = None, neuronios: int = None):
        super().__init__()
        self.camadas = camadas or PINN.numero_camadas
        self.neuronios = neuronios or PINN.neuronios_por_camada
        self.rede = RedeBasePINN(dim_entrada, dim_saida, self.camadas, self.neuronios)
        self.historico_perda = {"total": [], "dados": [], "fisica": [], "contorno": []}
        LOGGER.info(f"ArquiteturaPINN criada: {self.camadas}x{self.neuronios} - Luiz Tiago Wilcke")

    def forward(self, x):
        return self.rede(x)

    def perda_dados(self, pred: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        """L_data (Eq. Cap. 3.3.1)."""
        return torch.mean((pred - alvo)**2)

    def perda_fisica(self, residuo: torch.Tensor) -> torch.Tensor:
        """L_phys (Eq. Cap. 3.3.2)."""
        return torch.mean(residuo**2)

    def perda_contorno(self, pred_bc: torch.Tensor, alvo_bc: torch.Tensor) -> torch.Tensor:
        """L_bc (Eq. Cap. 3.3.3)."""
        return torch.mean((pred_bc - alvo_bc)**2)

    def perda_total(self, l_data: torch.Tensor, l_phys: torch.Tensor, l_bc: torch.Tensor,
                    w_data: float = None, w_phys: float = None, w_bc: float = None) -> torch.Tensor:
        w_data = w_data or PINN.peso_perda_dados
        w_phys = w_phys or PINN.peso_perda_fisica
        w_bc = w_bc or PINN.peso_perda_contorno
        return w_data * l_data + w_phys * l_phys + w_bc * l_bc

    def treinar_epoca(self, otimizador, pontos_col, pontos_bc, pontos_data=None, alvos_data=None):
        otimizador.zero_grad()
        # Exemplo simplificado de treino
        pred = self.forward(pontos_col)
        # Residuo dummy para exemplo
        residuo = pred * 0.0  # placeholder
        l_phys = self.perda_fisica(residuo)
        l_bc = torch.tensor(0.0)
        l_data = torch.tensor(0.0)
        if pontos_data is not None and alvos_data is not None:
            pred_d = self.forward(pontos_data)
            l_data = self.perda_dados(pred_d, alvos_data)
        perda = self.perda_total(l_data, l_phys, l_bc)
        perda.backward()
        otimizador.step()
        self.historico_perda["total"].append(perda.item())
        return perda.item()

class FrameworkMAP:
    """Equivalência com Estimativa de Máxima A Posteriori (Cap. 3.6)."""

    def __init__(self, prior_sigma: float = 1.0, likelihood_sigma: float = 0.1):
        self.prior_sigma = prior_sigma
        self.likelihood_sigma = likelihood_sigma

    def log_prior(self, pesos: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(pesos**2) / self.prior_sigma**2

    def log_likelihood(self, residuo: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(residuo**2) / self.likelihood_sigma**2

    def perda_map(self, pesos: torch.Tensor, residuo: torch.Tensor) -> torch.Tensor:
        return - (self.log_prior(pesos) + self.log_likelihood(residuo))
