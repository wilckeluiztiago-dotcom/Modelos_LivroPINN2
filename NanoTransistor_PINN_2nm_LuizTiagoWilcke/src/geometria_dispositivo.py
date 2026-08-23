"""
Módulo: Geometria do Nanotransistor de 2 nm
Autor: Luiz Tiago Wilcke
Descrição: Define geometria Double-Gate / GAA Nanosheet para canal de 2 nm.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class GeometriaNanotransistor:
    """Geometria física do dispositivo (unidades nm internamente)."""
    comprimento_canal_nm: float = 14.0      # L_g típico nó N2
    espessura_canal_nm: float = 2.0         # t_Si = 2 nm
    espessura_oxido_nm: float = 1.32        # EOT
    largura_nanosheet_nm: float = 15.0      # largura típica nanosheet
    numero_folhas: int = 3                  # stack de nanosheets
    temperatura_K: float = 300.0
    tipo: str = "GAA_nanosheet"             # ou "double_gate"

    def dominio_longitudinal(self, n_pontos: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """Malha 1D normalizada [0,1] → Fonte (0) até Dreno (1)."""
        x_norm = torch.linspace(0.0, 1.0, n_pontos)
        x_nm = x_norm * self.comprimento_canal_nm
        return x_norm, x_nm

    def dominio_transversal(self, n_pontos: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """Direção de confinamento quântico (espessura do canal)."""
        y_norm = torch.linspace(0.0, 1.0, n_pontos)
        y_nm = y_norm * self.espessura_canal_nm
        return y_norm, y_nm

    def dominio_2d(self, nx: int = 128, ny: int = 64):
        """Malha 2D (x,y) normalizada."""
        x_norm = torch.linspace(0.0, 1.0, nx)
        y_norm = torch.linspace(0.0, 1.0, ny)
        X, Y = torch.meshgrid(x_norm, y_norm, indexing="ij")
        return X, Y, x_norm * self.comprimento_canal_nm, y_norm * self.espessura_canal_nm

    def pontos_fronteira(self) -> Dict[str, float]:
        return {
            "fonte": 0.0,
            "dreno": 1.0,
            "interface_oxido_inf": 0.0,
            "interface_oxido_sup": 1.0,
            "centro_canal": 0.5,
        }

    def volume_efetivo_m3(self) -> float:
        """Volume aproximado do canal em m³."""
        L = self.comprimento_canal_nm * 1e-9
        t = self.espessura_canal_nm * 1e-9
        W = self.largura_nanosheet_nm * 1e-9 * self.numero_folhas
        return L * t * W

    def resumo(self) -> str:
        return (f"Geometria {self.tipo}: L={self.comprimento_canal_nm} nm, "
                f"t_Si={self.espessura_canal_nm} nm, EOT={self.espessura_oxido_nm} nm, "
                f"W={self.largura_nanosheet_nm} nm × {self.numero_folhas} folhas")


if __name__ == "__main__":
    geo = GeometriaNanotransistor()
    print(geo.resumo())
