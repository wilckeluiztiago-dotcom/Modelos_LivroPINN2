"""
Módulo 21: Pipeline Completo de Inferência e Produção
Autor: Luiz Tiago Wilcke

Permite carregar um modelo treinado e mapear qualquer curva forward
inicial f(0,·) para a superfície completa de preços P(t,T).
"""

import torch
from typing import Union
import numpy as np
from .config import CONFIG
from .arquitetura_deeponet import PIDeepONetHJM
from .exportacao import carregar_modelo
from .avaliacao import avaliar_superficie
from .geracao_curvas import curva_forward_nelson_siegel


class PipelineHJM:
    """
    Pipeline de produção do operador neural.

    Uso:
        pipe = PipelineHJM("results/modelo_pi_deeponet_hjm.pt")
        superficie = pipe.predizer_superficie(curva_forward)
    """

    def __init__(self, caminho_modelo: str = None):
        if caminho_modelo:
            self.modelo = carregar_modelo(caminho_modelo)
        else:
            self.modelo = PIDeepONetHJM()
            self.modelo.to(CONFIG.dispositivo)
            self.modelo.eval()

    @torch.no_grad()
    def predizer_preco(
        self,
        curva_forward: Union[np.ndarray, torch.Tensor],
        t: float,
        T: float,
    ) -> float:
        """Prediz P(t,T) para uma curva e um par (t,T)."""
        if isinstance(curva_forward, np.ndarray):
            u = torch.tensor(curva_forward, dtype=CONFIG.dtype, device=CONFIG.dispositivo).unsqueeze(0)
        else:
            u = curva_forward.to(CONFIG.dispositivo)
            if u.dim() == 1:
                u = u.unsqueeze(0)

        t_t = torch.tensor([[t]], dtype=CONFIG.dtype, device=CONFIG.dispositivo)
        T_t = torch.tensor([[T]], dtype=CONFIG.dtype, device=CONFIG.dispositivo)
        P = self.modelo(u, t_t, T_t)
        return P.item()

    def predizer_superficie(
        self,
        curva_forward: Union[np.ndarray, torch.Tensor],
        resolucao: int = 40,
    ) -> dict:
        """Retorna a superfície completa P(t,T)."""
        if isinstance(curva_forward, np.ndarray):
            u = torch.tensor(curva_forward, dtype=CONFIG.dtype, device=CONFIG.dispositivo).unsqueeze(0)
        else:
            u = curva_forward
        return avaliar_superficie(self.modelo, u, resolucao=resolucao)

    def exemplo_nelson_siegel(self) -> dict:
        """Gera uma curva NS de exemplo e a superfície correspondente."""
        T_sens = torch.tensor(CONFIG.maturidades_sensores, dtype=CONFIG.dtype, device=CONFIG.dispositivo)
        f0 = curva_forward_nelson_siegel(T_sens)
        return self.predizer_superficie(f0)
