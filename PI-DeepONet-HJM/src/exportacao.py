"""
Módulo 20: Exportação de Modelo e Resultados
Autor: Luiz Tiago Wilcke
"""

import torch
from pathlib import Path
from .arquitetura_deeponet import PIDeepONetHJM
from .config import CONFIG


def salvar_modelo(modelo: PIDeepONetHJM, caminho: str = "results/modelo.pt"):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "estado_modelo": modelo.state_dict(),
        "config": CONFIG.__dict__,
    }, caminho)
    print(f"Modelo salvo em {caminho}")


def carregar_modelo(caminho: str = "results/modelo.pt") -> PIDeepONetHJM:
    checkpoint = torch.load(caminho, map_location=CONFIG.dispositivo)
    modelo = PIDeepONetHJM()
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.to(CONFIG.dispositivo)
    modelo.eval()
    return modelo
