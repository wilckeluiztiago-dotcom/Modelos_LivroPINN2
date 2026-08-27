#!/usr/bin/env python3
# =============================================================================
# Script Principal - Modelo PINN para Poço de Petróleo
# Autor: Luiz Tiago Wilcke
# Baseado no livro: Redes Neurais Informadas pela Física - Volume 3
# =============================================================================
"""
Ponto de entrada do sistema.
Executa análise completa de geometria do poço, gera imagens e demonstra
módulos principais.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn_petroleo_wilcke.config.configuracoes import (
    FISICA, PINN, GEOMETRIA, SISTEMA, resumo_configuracoes, atualizar_configuracao_fisica
)
from pinn_petroleo_wilcke.utils.utilitarios import LOGGER, resumo_dimensoes_poco
from pinn_petroleo_wilcke.visualizacao.imagem_poco import GeradorImagemPoco, plotar_perfil_pressao
from pinn_petroleo_wilcke.modulos.modulo01_fundamentos import FundamentosReservatorio, RedeBasePINN
from pinn_petroleo_wilcke.modulos.modulo02_escoamento_vertical import EscoamentoVerticalMultifasico
from pinn_petroleo_wilcke.modulos.modulo03_arquitetura_pinn import ArquiteturaPINN

import numpy as np
import torch

def main():
    print("=" * 70)
    print("  MODELO PINN PARA POÇO DE PETRÓLEO")
    print("  Autor: Luiz Tiago Wilcke")
    print("  Volume 3: Dinâmica Multifásica, Elevação Artificial e Completações Inteligentes")
    print("=" * 70)
    print()
    print(resumo_configuracoes())
    print()

    # 1. Análise de dimensões do poço
    print(">>> 1. Análise de Tamanho e Dimensões do Poço")
    dimensoes = resumo_dimensoes_poco(
        GEOMETRIA.profundidade_medida,
        GEOMETRIA.diametro_revestimento,
        GEOMETRIA.diametro_tubing,
        FISICA.espessura_reservatorio
    )
    for k, v in dimensoes.items():
        print(f"    {k}: {v}")
    print()

    # 2. Geração de imagens do poço
    print(">>> 2. Gerando Imagens Esquemáticas do Poço...")
    gerador = GeradorImagemPoco()
    relatorio = gerador.analisar_tamanho_e_gerar_relatorio()
    print(f"    Imagens geradas: {len(relatorio.get('imagens_geradas', []))}")
    for img in relatorio.get("imagens_geradas", []):
        print(f"      - {img}")
    print()

    # 3. Fundamentos físicos
    print(">>> 3. Fundamentos de Reservatório")
    fund = FundamentosReservatorio()
    resumo_fund = fund.resumo()
    for k, v in resumo_fund.items():
        print(f"    {k}: {v}")
    print()

    # Solução analítica radial
    r = np.logspace(np.log10(FISICA.raio_poco), np.log10(FISICA.raio_drenagem), 100)
    p_wf = 15e6
    p_e = FISICA.pressao_inicial
    p = fund.pressao_radial_estacionaria(r, p_wf, p_e)
    print(f"    Pressão no fundo (Pwf): {p_wf/1e6:.2f} MPa")
    print(f"    Pressão externa (Pe): {p_e/1e6:.2f} MPa")
    print(f"    Vazão Dupuit: {fund.vazao_dupuit(p_e, p_wf)*86400:.1f} m³/d")
    plotar_perfil_pressao(r, p)
    print("    Perfil de pressão salvo em figuras/")
    print()

    # 4. Escoamento vertical
    print(">>> 4. Escoamento Vertical Multifásico")
    esc = EscoamentoVerticalMultifasico()
    dens_mist = esc.densidade_mistura_holdup(0.7, FISICA.densidade_oleo, FISICA.densidade_gas)
    dp_dz = esc.gradiente_pressao_total(dens_mist, 1.5, GEOMETRIA.diametro_tubing, FISICA.viscosidade_oleo)
    print(f"    Densidade mistura: {dens_mist:.1f} kg/m³")
    print(f"    Gradiente de pressão total: {dp_dz/1e6:.4f} MPa/m")
    print(f"    Padrão de fluxo (exemplo): {esc.padrao_escoamento(0.5, 2.0, GEOMETRIA.diametro_tubing)}")
    print()

    # 5. Demonstração PINN
    print(">>> 5. Arquitetura PINN (demonstração)")
    modelo = ArquiteturaPINN(dim_entrada=2, dim_saida=1)
    print(f"    Rede criada: {PINN.numero_camadas} camadas x {PINN.neuronios_por_camada} neurônios")
    print(f"    Dispositivo: {PINN.dispositivo}")
    # Forward pass dummy
    x = torch.rand(10, 2)
    y = modelo(x)
    print(f"    Forward pass OK - shape saída: {y.shape}")
    print()

    print("=" * 70)
    print("  EXECUÇÃO CONCLUÍDA COM SUCESSO")
    print("  Autor: Luiz Tiago Wilcke")
    print("  Todos os 25 módulos disponíveis em pinn_petroleo_wilcke/modulos/")
    print("=" * 70)

if __name__ == "__main__":
    main()
