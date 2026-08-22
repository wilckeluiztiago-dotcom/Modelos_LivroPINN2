# -*- coding: utf-8 -*-
"""
Módulo 29: Logger TensorBoard em Tempo Real
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
from typing import Dict
import torch

class MonitorTensorBoard:
    def __init__(self, log_dir: str = "runs/set_pinn"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.ativo = True
        except Exception:
            self.writer = None
            self.ativo = False
            print("[Aviso] TensorBoard não disponível. Continuando sem logger.")

    def registrar(self, perdas: Dict[str, float], epoch: int) -> None:
        if not self.ativo:
            return
        for nome, valor in perdas.items():
            self.writer.add_scalar(f"Perda/{nome}", valor, epoch)

    def fechar(self) -> None:
        if self.ativo and self.writer is not None:
            self.writer.close()
