"""
Módulo 13: Loop de Treinamento Híbrido
Autor: Luiz Tiago Wilcke
"""

import torch
from tqdm import tqdm
from .config import CONFIG
from .arquitetura_deeponet import PIDeepONetHJM
from .perda_composta import perda_composta
from .amostragem import amostrar_dominio
from .otimizacao import criar_otimizador_adam, criar_otimizador_lbfgs
from .logger import Logger


def treinar(
    modelo: PIDeepONetHJM,
    num_epocas_adam: int = CONFIG.num_epocas_adam,
    num_epocas_lbfgs: int = CONFIG.num_epocas_lbfgs,
    logger: Logger = None,
) -> dict:
    """
    Treinamento em duas fases: Adam (rápido) + L-BFGS (refinamento).
    """
    modelo.to(CONFIG.dispositivo)
    otim_adam = criar_otimizador_adam(modelo)
    historico = {"total": [], "fisica": [], "dados": []}

    # Fase 1: Adam
    pbar = tqdm(range(num_epocas_adam), desc="Adam")
    for epoca in pbar:
        batch = amostrar_dominio()
        T_sensores = batch["T_sensores"]

        otim_adam.zero_grad()
        perdas = perda_composta(modelo, batch, T_sensores)
        perdas["total"].backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otim_adam.step()

        historico["total"].append(perdas["total"].item())
        historico["fisica"].append(perdas["fisica"].item())
        historico["dados"].append(perdas["dados"].item())

        if epoca % 100 == 0:
            pbar.set_postfix({
                "loss": f"{perdas['total'].item():.2e}",
                "fis": f"{perdas['fisica'].item():.2e}",
            })
            if logger:
                logger.registrar(epoca, perdas)

    # Fase 2: L-BFGS
    if num_epocas_lbfgs > 0:
        otim_lbfgs = criar_otimizador_lbfgs(modelo)

        def closure():
            otim_lbfgs.zero_grad()
            batch = amostrar_dominio(num_pontos=2048)
            perdas = perda_composta(modelo, batch, batch["T_sensores"])
            perdas["total"].backward()
            return perdas["total"]

        pbar2 = tqdm(range(num_epocas_lbfgs), desc="L-BFGS")
        for epoca in pbar2:
            loss = otim_lbfgs.step(closure)
            historico["total"].append(loss if isinstance(loss, float) else loss.item())
            pbar2.set_postfix({"loss": f"{historico['total'][-1]:.2e}"})

    return historico
