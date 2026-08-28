"""
Módulo 02: Utilitários Gerais
Autor: Luiz Tiago Wilcke
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .config import CONFIG


def definir_semente(seed: int = CONFIG.seed) -> None:
    """Fixa a semente para reprodutibilidade."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def para_tensor(x, dispositivo: Optional[str] = None, requer_grad: bool = False) -> torch.Tensor:
    """Converte array/list para tensor no dispositivo correto."""
    dispositivo = dispositivo or CONFIG.dispositivo
    if isinstance(x, torch.Tensor):
        t = x.to(dispositivo=dispositivo, dtype=CONFIG.dtype)
    else:
        t = torch.tensor(x, dtype=CONFIG.dtype, device=dispositivo)
    if requer_grad:
        t.requires_grad_(True)
    return t


def normalizar_maturidade(T: torch.Tensor, T_min: float = CONFIG.T_min, T_max: float = CONFIG.T_max) -> torch.Tensor:
    """Normaliza maturidade para [0, 1]."""
    return (T - T_min) / (T_max - T_min + 1e-8)


def desnormalizar_maturidade(T_norm: torch.Tensor, T_min: float = CONFIG.T_min, T_max: float = CONFIG.T_max) -> torch.Tensor:
    """Desnormaliza maturidade."""
    return T_norm * (T_max - T_min) + T_min


def taxa_curta_de_forward(f_tT: torch.Tensor, t: torch.Tensor, T: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """Extrai r(t) = f(t, t) por interpolação quando T ≈ t."""
    mascara = torch.abs(T - t) < tol
    return torch.where(mascara, f_tT, torch.zeros_like(f_tT))


def produto_interno(branch: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
    """Produto interno b(u) · t(y) + bias."""
    return torch.sum(branch * trunk, dim=-1, keepdim=True)
