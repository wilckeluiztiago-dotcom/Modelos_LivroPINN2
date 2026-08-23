"""
Módulo: Mean-Field Games (analogia coletiva de variação de processo)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn


class MFGVariacaoProcesso(nn.Module):
    """
    Modelo simplificado de Mean-Field Game para distribuição de
    parâmetros de dispositivo em uma população de chips.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.hjb_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.fp_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def valor(self, estado):
        return self.hjb_net(estado)

    def densidade(self, estado):
        return torch.softmax(self.fp_net(estado), dim=0)
