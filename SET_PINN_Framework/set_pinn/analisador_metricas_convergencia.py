# -*- coding: utf-8 -*-
"""
Módulo 30: Métricas Formais de Convergência (L2, RMSE, Resíduo Máximo)
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import DTYPE

def metricas_erro(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = pred - ref
    l2 = torch.norm(diff) / (torch.norm(ref) + 1e-12)
    rmse = torch.sqrt(torch.mean(diff**2))
    max_res = torch.max(torch.abs(diff))
    return {
        "L2_relativo": float(l2.item()),
        "RMSE": float(rmse.item()),
        "Residuo_Maximo": float(max_res.item())
    }
