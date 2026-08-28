"""
Módulo 19: Script Principal de Execução
Autor: Luiz Tiago Wilcke
Projeto: PI-DeepONet para Dinâmica da Curva de Juros (HJM)
"""

import torch
from .config import CONFIG
from .utils import definir_semente
from .arquitetura_deeponet import PIDeepONetHJM
from .treinamento import treinar
from .avaliacao import erro_relativo_medio
from .visualizacao import plotar_superficie_precos, plotar_historico_perda, plotar_curva_forward_inicial
from .geracao_curvas import gerar_ensemble_curvas
from .logger import Logger
from .exportacao import salvar_modelo


def main():
    print("=" * 60)
    print("PI-DeepONet HJM – Operador Neural para Curva de Juros")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)

    definir_semente(CONFIG.seed)
    print(f"Dispositivo: {CONFIG.dispositivo}")

    # Modelo
    modelo = PIDeepONetHJM(CONFIG)
    num_params = sum(p.numel() for p in modelo.parameters())
    print(f"Parâmetros treináveis: {num_params:,}")

    # Logger
    logger = Logger()

    # Treinamento
    historico = treinar(modelo, logger=logger)

    # Avaliação
    erro = erro_relativo_medio(modelo)
    print(f"Erro relativo médio do residual EDP: {erro:.4e}")

    # Visualizações
    u, T_sens = gerar_ensemble_curvas(num_curvas=1)
    plotar_curva_forward_inicial(u[0].cpu().numpy(), T_sens.cpu().numpy())
    plotar_superficie_precos(modelo, u)
    plotar_historico_perda(historico)

    # Exportação
    salvar_modelo(modelo, "results/modelo_pi_deeponet_hjm.pt")
    print("Treinamento concluído com sucesso.")


if __name__ == "__main__":
    main()
