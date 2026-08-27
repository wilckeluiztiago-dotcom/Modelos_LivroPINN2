# =============================================================================
# REDES NEURAIS INFORMADAS PELA FÍSICA - VOLUME 3
# Engenharia de Petróleo e Poços
# Autor: Luiz Tiago Wilcke
# Especialista em Deep Learning Científico e Engenharia de Petróleo
# VOLUME III: COMPUTAÇÃO NEURAL APLICADA À SUBSUPERFÍCIE
# =============================================================================
"""
Pacote principal de Modelagem PINN para Poços de Petróleo.
Baseado no livro "Redes Neurais Informadas pela Física - Volume 3".

Este sistema modular implementa 25 módulos especializados para:
- Física de reservatórios e meios porosos
- Escoamento multifásico vertical e horizontal
- Elevação artificial (Gas Lift, BCS, Bombeio Mecânico)
- Completações inteligentes e ICDs
- Problemas inversos e ajuste de histórico
- Fluidos não-newtonianos
- Geomecânica e fraturamento hidráulico
- Operadores neurais (DeepONet, FNO, PINO)
- Severe slugging em risers
- XPINNs e decomposição de domínio
- Acoplamento THMC
- Aprendizado por reforço informado pela física (PIRL)
- Aplicações no Pré-Sal brasileiro
- Termodinâmica composicional HPHT
- Fluência de sal, wormholes, drift-flux, swelling de folhelhos
- Tomografia eletromagnética e inversão sísmica
- Armazenamento subterrâneo de hidrogênio (UHS)

Variáveis e nomes em Português conforme o livro.
"""

__version__ = "3.0.0"
__author__ = "Luiz Tiago Wilcke"
__email__ = "contato@wilcke-petroleo.ai"
__livro__ = "Redes Neurais Informadas pela Física - Volume 3: Dinâmica Multifásica, Sistemas de Elevação Artificial e Completações Inteligentes"

from .modulos import *
from .utils import *
from .visualizacao import *
from .config import *

print(f"PINN Petróleo Wilcke v{__version__} carregado com sucesso.")
print(f"Autor: {__author__}")
print(f"Baseado em: {__livro__}")
