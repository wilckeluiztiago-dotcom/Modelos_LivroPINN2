"""
Módulo 01: Geometria do Nanotransistor de 2 nm
Autor: Luiz Tiago Wilcke
Descrição: Define a geometria Double-Gate / GAA simplificada 1D/2D para canal de 2 nm.
"""

import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class GeometriaNanotransistor:
    """Geometria física do dispositivo (unidades SI → nm internamente convertidos)."""
    comprimento_canal_nm: float = 14.0      # L_g típico N2
    espessura_canal_nm: float = 2.0         # t_si = 2 nm (foco do estudo)
    espessura_oxido_nm: float = 1.32        # EOT
    largura_nm: float = 10.0                # para GAA/nanosheet
    temperatura_K: float = 300.0

    def dominio_espacial(self, n_pontos: int = 1000):
        """Retorna malha 1D normalizada [0,1] mapeada para o canal."""
        x = torch.linspace(0.0, 1.0, n_pontos)
        # mapeamento físico: 0 → source, 1 → drain
        x_fisico_nm = x * self.comprimento_canal_nm
        return x, x_fisico_nm

    def dominio_transversal(self, n_pontos: int = 200):
        """Direção de confinamento (espessura 2 nm)."""
        y = torch.linspace(0.0, 1.0, n_pontos)
        y_fisico_nm = y * self.espessura_canal_nm
        return y, y_fisico_nm

    def pontos_fronteira(self):
        """Pontos de contorno para BC de potencial e onda."""
        return {
            "fonte": 0.0,
            "dreno": 1.0,
            "interface_oxido_inferior": 0.0,
            "interface_oxido_superior": 1.0
        }

if __name__ == "__main__":
    geo = GeometriaNanotransistor()
    print(f"Geometria carregada: canal {geo.espessura_canal_nm} nm × {geo.comprimento_canal_nm} nm")
