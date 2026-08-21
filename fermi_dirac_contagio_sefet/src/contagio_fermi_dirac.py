"""
Dinâmica de contágio com probabilidades de transição Fermi–Dirac.

    P_{ij} = 1 / (1 + exp(β (D_i − D_j)))

Apêndice J.4.
"""

import numpy as np
from typing import Optional, Tuple
from .cadeia_dopantes import CadeiaDopantes


def probabilidade_fermi_dirac(
    D_i: float,
    D_j: float,
    beta: float = 2.0,
) -> float:
    """
    Probabilidade bilateral de salto j → i (elétron em j tenta ir para i):
        P_{ij} = 1 / (1 + e^{β(D_i − D_j)})

    Quando D_i < D_j (destino mais baixo), P → 1 (favorável).
    """
    return 1.0 / (1.0 + np.exp(beta * (D_i - D_j)))


def passo_contagio(
    cadeia: CadeiaDopantes,
    beta: float = 2.0,
    taxa_source: float = 0.3,
    taxa_drain: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Um passo de dinâmica estocástica:

    1) Injeção source → sítio 0 (se vazio)
    2) Saltos entre sítios vizinhos com P_ij Fermi–Dirac
    3) Ejeção sítio final → drain (se ocupado)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = cadeia.n_sitios
    eventos = {"injecao": 0, "salto": 0, "ejecao": 0}

    # injeção source
    if cadeia.ocupacao[0] == 0 and rng.random() < taxa_source:
        # energia do "reservatório" source ≈ V_source
        D_dest = cadeia.energia_sitio(0)
        P = probabilidade_fermi_dirac(D_dest, cadeia.V_source, beta)
        if rng.random() < P:
            cadeia.ocupacao[0] = 1
            eventos["injecao"] = 1

    # saltos vizinhos (varredura aleatória)
    ordem = rng.permutation(n - 1)
    for i in ordem:
        j = i + 1
        # elétron em i → j
        if cadeia.ocupacao[i] == 1 and cadeia.ocupacao[j] == 0:
            Di = cadeia.energia_sitio(i)
            Dj = cadeia.energia_sitio(j)
            P = probabilidade_fermi_dirac(Dj, Di, beta)  # destino j, origem i
            if rng.random() < P:
                cadeia.ocupacao[i] = 0
                cadeia.ocupacao[j] = 1
                eventos["salto"] = 1
                break
        # elétron em j → i
        elif cadeia.ocupacao[j] == 1 and cadeia.ocupacao[i] == 0:
            Di = cadeia.energia_sitio(i)
            Dj = cadeia.energia_sitio(j)
            P = probabilidade_fermi_dirac(Di, Dj, beta)
            if rng.random() < P:
                cadeia.ocupacao[j] = 0
                cadeia.ocupacao[i] = 1
                eventos["salto"] = 1
                break

    # ejeção drain
    if cadeia.ocupacao[-1] == 1 and rng.random() < taxa_drain:
        D_orig = cadeia.energia_sitio(n - 1)
        P = probabilidade_fermi_dirac(cadeia.V_drain, D_orig, beta)
        if rng.random() < P:
            cadeia.ocupacao[-1] = 0
            eventos["ejecao"] = 1

    return eventos


def simular_transporte(
    cadeia: CadeiaDopantes,
    n_passos: int = 5000,
    beta: float = 2.0,
    taxa_source: float = 0.35,
    taxa_drain: float = 0.35,
    semente: Optional[int] = 0,
) -> dict:
    """
    Simula transporte elétron-a-elétron e registra:
    - corrente (ejeções)
    - ocupação média
    - número de elétrons no tempo
    """
    g = np.random.default_rng(semente)
    corrente = np.zeros(n_passos)
    n_el = np.zeros(n_passos)
    ocup_media = np.zeros((n_passos, cadeia.n_sitios))

    for t in range(n_passos):
        ev = passo_contagio(cadeia, beta, taxa_source, taxa_drain, g)
        corrente[t] = ev["ejecao"]
        n_el[t] = cadeia.numero_eletrons()
        ocup_media[t] = cadeia.ocupacao.copy()

    return {
        "corrente": corrente,
        "n_eletrons": n_el,
        "ocupacao": ocup_media,
        "corrente_media": float(np.mean(corrente)),
    }
