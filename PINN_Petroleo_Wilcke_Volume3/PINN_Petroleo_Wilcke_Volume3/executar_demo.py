#!/usr/bin/env python3
"""
Script de demonstração do Modelo PINN para Poço de Petróleo
Autor: Luiz Tiago Wilcke
"""
import sys
import os

# Garante que o pacote seja encontrado
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Ajusta imports relativos simulando pacote
import config.configuracoes as configuracoes
from config.configuracoes import FISICA, PINN, GEOMETRIA, SISTEMA, resumo_configuracoes
from utils.utilitarios import resumo_dimensoes_poco, LOGGER
from visualizacao.imagem_poco import GeradorImagemPoco, plotar_perfil_pressao

# Importa fundamentos com ajuste de path
sys.path.insert(0, os.path.join(ROOT, "modulos"))
# Como os módulos usam relative imports (..), precisamos de estrutura de pacote
# Solução: executar via python -m a partir do diretório pai

print("=" * 70)
print("  MODELO PINN PARA POÇO DE PETRÓLEO - DEMONSTRAÇÃO")
print("  Autor: Luiz Tiago Wilcke")
print("  Volume 3: Dinâmica Multifásica, Elevação Artificial e Completações Inteligentes")
print("=" * 70)
print()
print(resumo_configuracoes())
print()

print(">>> Análise de Dimensões do Poço")
dim = resumo_dimensoes_poco(
    GEOMETRIA.profundidade_medida,
    GEOMETRIA.diametro_revestimento,
    GEOMETRIA.diametro_tubing,
    FISICA.espessura_reservatorio
)
for k, v in dim.items():
    print(f"  {k}: {v}")
print()

print(">>> Gerando imagens do poço...")
try:
    import matplotlib
    matplotlib.use("Agg")
    g = GeradorImagemPoco()
    rel = g.analisar_tamanho_e_gerar_relatorio()
    print(f"  Imagens salvas em: {SISTEMA.diretorio_figuras}")
    for img in rel.get("imagens_geradas", []):
        print(f"    - {os.path.basename(img)}")
except Exception as e:
    print(f"  (imagens: {e})")

print()
print("=" * 70)
print("  Demonstração concluída com sucesso!")
print("  Autor: Luiz Tiago Wilcke")
print("  25 módulos disponíveis em ./modulos/")
print("=" * 70)
