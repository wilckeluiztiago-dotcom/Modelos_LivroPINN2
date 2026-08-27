# =============================================================================
# Utilitários Gerais do Sistema PINN
# Autor: Luiz Tiago Wilcke
# =============================================================================
"""
Funções auxiliares: conversões de unidades, logging, validação de dados,
cálculos petrofísicos básicos e helpers de tensor.
"""

import numpy as np
import torch
import logging
from typing import Union, Tuple, Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def configurar_logger(nome: str = "pinn_petroleo", nivel: str = "INFO") -> logging.Logger:
    """Configura logger padronizado."""
    logger = logging.getLogger(nome)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formato = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formato)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    return logger

LOGGER = configurar_logger()

# -----------------------------------------------------------------------------
# Conversões de Unidades
# -----------------------------------------------------------------------------
def darcy_para_m2(permeabilidade_md: float) -> float:
    """Converte milidarcy para m²."""
    return permeabilidade_md * 9.869233e-16

def m2_para_darcy(permeabilidade_m2: float) -> float:
    """Converte m² para milidarcy."""
    return permeabilidade_m2 / 9.869233e-16

def psi_para_pascal(pressao_psi: float) -> float:
    """Converte psi para Pascal."""
    return pressao_psi * 6894.757

def pascal_para_psi(pressao_pa: float) -> float:
    """Converte Pascal para psi."""
    return pressao_pa / 6894.757

def celsius_para_kelvin(temp_c: float) -> float:
    return temp_c + 273.15

def kelvin_para_celsius(temp_k: float) -> float:
    return temp_k - 273.15

def grau_api_para_densidade(grau_api: float, densidade_agua: float = 1000.0) -> float:
    """Calcula densidade do óleo a partir do °API (Eq. 1.1 do livro)."""
    sg = 141.5 / (grau_api + 131.5)
    return sg * densidade_agua

def densidade_para_grau_api(densidade: float, densidade_agua: float = 1000.0) -> float:
    sg = densidade / densidade_agua
    return 141.5 / sg - 131.5

# -----------------------------------------------------------------------------
# Cálculos Petrofísicos Básicos
# -----------------------------------------------------------------------------
def calcular_porosidade_pressao(porosidade_0: float, cf: float, pressao: float, pressao_0: float) -> float:
    """Porosidade compressível (Eq. 1.6)."""
    return porosidade_0 * np.exp(cf * (pressao - pressao_0))

def calcular_compressibilidade_total(cl: float, cf: float) -> float:
    """Compressibilidade total ct = cl + cf."""
    return cl + cf

def calcular_velocidade_darcy(permeabilidade: float, viscosidade: float,
                              gradiente_pressao: float, densidade: float = 0.0,
                              gravidade: float = 9.81) -> float:
    """Lei de Darcy unidimensional (Eq. 1.8)."""
    return - (permeabilidade / viscosidade) * (gradiente_pressao - densidade * gravidade)

def calcular_raio_efetivo(rw: float, s: float = 0.0) -> float:
    """Raio efetivo do poço com fator de pele."""
    return rw * np.exp(-s)

def calcular_indice_produtividade(k: float, h: float, mu: float, re: float, rw: float,
                                  s: float = 0.0, b: float = 1.0) -> float:
    """Índice de Produtividade (IP) radial estacionário."""
    return (2 * np.pi * k * h) / (mu * b * (np.log(re / rw) + s))

# -----------------------------------------------------------------------------
# Helpers de Tensor / PyTorch
# -----------------------------------------------------------------------------
def para_tensor(dados: Union[np.ndarray, list, float], dispositivo: str = "cpu",
                requer_grad: bool = False) -> torch.Tensor:
    """Converte dados para tensor PyTorch."""
    if isinstance(dados, torch.Tensor):
        t = dados.to(dispositivo)
    else:
        t = torch.tensor(np.asarray(dados), dtype=torch.float32, device=dispositivo)
    if requer_grad:
        t.requires_grad_(True)
    return t

def gradiente_autograd(saida: torch.Tensor, entrada: torch.Tensor,
                       criar_grafo: bool = True) -> torch.Tensor:
    """Calcula gradiente via Autograd de forma segura."""
    grad = torch.autograd.grad(
        outputs=saida,
        inputs=entrada,
        grad_outputs=torch.ones_like(saida),
        create_graph=criar_grafo,
        retain_graph=True,
        allow_unused=True
    )[0]
    if grad is None:
        return torch.zeros_like(entrada)
    return grad

def segunda_derivada(saida: torch.Tensor, entrada: torch.Tensor) -> torch.Tensor:
    """Calcula segunda derivada (∂²u/∂x²)."""
    primeira = gradiente_autograd(saida, entrada)
    return gradiente_autograd(primeira, entrada)

# -----------------------------------------------------------------------------
# Validação e Persistência
# -----------------------------------------------------------------------------
def validar_saturacao(saturacao: Union[float, np.ndarray, torch.Tensor]) -> bool:
    """Valida que saturação está entre 0 e 1."""
    if isinstance(saturacao, torch.Tensor):
        return bool(torch.all((saturacao >= 0) & (saturacao <= 1)))
    return bool(np.all((np.asarray(saturacao) >= 0) & (np.asarray(saturacao) <= 1)))

def salvar_checkpoint(modelo: torch.nn.Module, otimizador: torch.optim.Optimizer,
                      epoca: int, perda: float, caminho: str) -> None:
    """Salva checkpoint de treinamento."""
    estado = {
        "epoca": epoca,
        "estado_modelo": modelo.state_dict(),
        "estado_otimizador": otimizador.state_dict(),
        "perda": perda,
        "timestamp": datetime.now().isoformat(),
        "autor": "Luiz Tiago Wilcke"
    }
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    torch.save(estado, caminho)
    LOGGER.info(f"Checkpoint salvo: {caminho}")

def carregar_checkpoint(modelo: torch.nn.Module, otimizador: torch.optim.Optimizer,
                        caminho: str) -> Tuple[int, float]:
    """Carrega checkpoint."""
    estado = torch.load(caminho, map_location="cpu")
    modelo.load_state_dict(estado["estado_modelo"])
    otimizador.load_state_dict(estado["estado_otimizador"])
    LOGGER.info(f"Checkpoint carregado da época {estado['epoca']}")
    return estado["epoca"], estado["perda"]

def salvar_json(dados: Dict, caminho: str) -> None:
    """Salva dicionário como JSON."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)

def carregar_json(caminho: str) -> Dict:
    """Carrega JSON."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------------------------------------------------------
# Análise de Tamanho do Poço
# -----------------------------------------------------------------------------
def calcular_volume_poco(diametro: float, comprimento: float) -> float:
    """Volume interno do poço [m³]."""
    raio = diametro / 2.0
    return np.pi * raio**2 * comprimento

def calcular_area_lateral(diametro: float, comprimento: float) -> float:
    """Área lateral da parede do poço [m²]."""
    return np.pi * diametro * comprimento

def resumo_dimensoes_poco(profundidade_md: float, diametro_revestimento: float,
                          diametro_tubing: float, espessura: float = 30.0) -> Dict:
    """Gera resumo completo de dimensões do poço."""
    volume_revestimento = calcular_volume_poco(diametro_revestimento, profundidade_md)
    volume_tubing = calcular_volume_poco(diametro_tubing, profundidade_md)
    area_lateral = calcular_area_lateral(diametro_revestimento, profundidade_md)
    volume_reservatorio_aproximado = np.pi * (500.0)**2 * espessura  # re=500m
    return {
        "profundidade_md_m": profundidade_md,
        "diametro_revestimento_m": diametro_revestimento,
        "diametro_tubing_m": diametro_tubing,
        "volume_interno_revestimento_m3": volume_revestimento,
        "volume_interno_tubing_m3": volume_tubing,
        "area_lateral_m2": area_lateral,
        "volume_reservatorio_aproximado_m3": volume_reservatorio_aproximado,
        "razao_volume_poco_reservatorio": volume_revestimento / volume_reservatorio_aproximado,
        "autor": "Luiz Tiago Wilcke"
    }
