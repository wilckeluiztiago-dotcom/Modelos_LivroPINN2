# =============================================================================
# Módulo de Configurações Globais
# Autor: Luiz Tiago Wilcke
# =============================================================================
"""
Configurações centrais do sistema PINN para análise de poços de petróleo.
Inclui parâmetros físicos, hiperparâmetros de treino, caminhos e constantes.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class ConfiguracaoFisica:
    """Parâmetros físicos padrão baseados no livro (Capítulo 1)."""
    porosidade_referencia: float = 0.20          # φ0
    compressibilidade_poros: float = 1.0e-9      # cf [1/Pa]
    compressibilidade_fluido: float = 1.0e-9     # cl [1/Pa]
    viscosidade_oleo: float = 0.005              # μo [Pa.s] ~ 5 cP
    viscosidade_agua: float = 0.001              # μw [Pa.s]
    viscosidade_gas: float = 1.5e-5              # μg [Pa.s]
    densidade_oleo: float = 850.0                # ρo [kg/m³]
    densidade_agua: float = 1000.0               # ρw [kg/m³]
    densidade_gas: float = 80.0                  # ρg [kg/m³] (padrão)
    permeabilidade_horizontal: float = 100.0e-15 # kh [m²] ~ 100 mD
    permeabilidade_vertical: float = 10.0e-15    # kv [m²]
    raio_poco: float = 0.108                     # rw [m] ~ 4.25"
    raio_drenagem: float = 500.0                 # re [m]
    espessura_reservatorio: float = 30.0         # h [m]
    pressao_inicial: float = 25.0e6              # Pi [Pa]
    temperatura_reservatorio: float = 85.0       # T [°C]
    grau_api: float = 28.0                       # °API
    fator_acentrico: float = 0.3                 # ω (Pitzer)
    pressao_critica: float = 4.5e6               # Pc [Pa]
    temperatura_critica: float = 450.0           # Tc [K]

@dataclass
class ConfiguracaoPINN:
    """Hiperparâmetros de treinamento das PINNs."""
    numero_camadas: int = 8
    neuronios_por_camada: int = 64
    taxa_aprendizado: float = 1e-3
    numero_epocas: int = 5000
    tamanho_lote: int = 1024
    pontos_colocacao: int = 10000
    pontos_contorno: int = 2000
    pontos_dados: int = 500
    peso_perda_dados: float = 1.0
    peso_perda_fisica: float = 1.0
    peso_perda_contorno: float = 10.0
    ativacao: str = "tanh"
    otimizador: str = "adam"
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"
    semente_aleatoria: int = 42
    usar_autograd: bool = True
    regularizacao_l2: float = 1e-6
    tolerancia_convergencia: float = 1e-6

@dataclass
class ConfiguracaoGeometriaPoco:
    """Parâmetros geométricos do poço para análise de tamanho e imagem."""
    tipo_poco: str = "vertical"                  # vertical, horizontal, direcional, multilateral
    profundidade_medida: float = 3500.0          # MD [m]
    profundidade_vertical: float = 3200.0        # TVD [m]
    diametro_revestimento: float = 0.244         # [m] 9 5/8"
    diametro_tubing: float = 0.114               # [m] 4 1/2"
    diametro_open_hole: float = 0.216            # [m] 8 1/2"
    inclinacao: float = 0.0                      # [graus]
    azimuth: float = 0.0                         # [graus]
    numero_segmentos: int = 20
    comprimento_horizontal: float = 1000.0       # para horizontais
    numero_laterais: int = 1                     # para multilaterais
    tipo_completacao: str = "cimentada"          # cimentada, slotted_liner, intelligent, open_hole
    sensores_dts: bool = False
    sensores_das: bool = False
    valvulas_icv: int = 0
    dispositivos_icd: int = 0

@dataclass
class ConfiguracaoSistema:
    """Configurações gerais do sistema."""
    diretorio_raiz: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diretorio_dados: str = field(default="")
    diretorio_resultados: str = field(default="")
    diretorio_figuras: str = field(default="")
    nivel_log: str = "INFO"
    salvar_checkpoints: bool = True
    intervalo_checkpoint: int = 500
    gerar_imagens: bool = True
    formato_imagem: str = "png"
    dpi_imagem: int = 150
    idioma: str = "pt-BR"
    autor: str = "Luiz Tiago Wilcke"
    versao: str = "3.0.0"

    def __post_init__(self):
        self.diretorio_dados = os.path.join(self.diretorio_raiz, "dados")
        self.diretorio_resultados = os.path.join(self.diretorio_raiz, "resultados")
        self.diretorio_figuras = os.path.join(self.diretorio_raiz, "figuras")
        for d in [self.diretorio_dados, self.diretorio_resultados, self.diretorio_figuras]:
            os.makedirs(d, exist_ok=True)

# Instâncias globais
FISICA = ConfiguracaoFisica()
PINN = ConfiguracaoPINN()
GEOMETRIA = ConfiguracaoGeometriaPoco()
SISTEMA = ConfiguracaoSistema()

def atualizar_configuracao_fisica(**kwargs):
    """Atualiza parâmetros físicos dinamicamente."""
    global FISICA
    for chave, valor in kwargs.items():
        if hasattr(FISICA, chave):
            setattr(FISICA, chave, valor)
        else:
            raise AttributeError(f"Parâmetro físico desconhecido: {chave}")

def atualizar_configuracao_pinn(**kwargs):
    """Atualiza hiperparâmetros PINN."""
    global PINN
    for chave, valor in kwargs.items():
        if hasattr(PINN, chave):
            setattr(PINN, chave, valor)

def obter_dispositivo() -> torch.device:
    """Retorna o dispositivo de computação."""
    return torch.device(PINN.dispositivo)

def resumo_configuracoes() -> str:
    """Gera resumo textual das configurações."""
    resumo = []
    resumo.append("=" * 60)
    resumo.append(f"CONFIGURAÇÕES PINN PETRÓLEO - Autor: {SISTEMA.autor}")
    resumo.append("=" * 60)
    resumo.append(f"Tipo de Poço: {GEOMETRIA.tipo_poco}")
    resumo.append(f"Profundidade MD: {GEOMETRIA.profundidade_medida} m")
    resumo.append(f"Raio do Poço: {FISICA.raio_poco} m")
    resumo.append(f"Porosidade: {FISICA.porosidade_referencia}")
    resumo.append(f"Permeabilidade H: {FISICA.permeabilidade_horizontal*1e15:.1f} mD")
    resumo.append(f"Camadas PINN: {PINN.numero_camadas} x {PINN.neuronios_por_camada}")
    resumo.append(f"Dispositivo: {PINN.dispositivo}")
    resumo.append("=" * 60)
    return "\n".join(resumo)
