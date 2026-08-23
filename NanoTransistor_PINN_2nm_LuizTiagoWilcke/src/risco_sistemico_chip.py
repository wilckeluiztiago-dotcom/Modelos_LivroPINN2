"""
Módulo: Risco Sistêmico de Variação de Processo (analogia redes interbancárias)
Autor: Luiz Tiago Wilcke
"""

import torch


class RiscoSistemicoChip:
    """
    Modelo de contágio de falha entre dispositivos em um chip
    (analogia a redes interbancárias do livro de PINNs financeiras).
    """
    def __init__(self, n_dispositivos=100, limiar=0.3):
        self.n = n_dispositivos
        self.limiar = limiar

    def matriz_acoplamento(self, correlacao=0.2):
        A = correlacao * torch.ones(self.n, self.n)
        A.fill_diagonal_(0.0)
        return A

    def propagacao_falha(self, estados, A):
        """estados ∈ [0,1]; 1 = falha."""
        for _ in range(10):
            influencia = A @ estados
            novos = (influencia > self.limiar).float()
            if torch.allclose(novos, estados):
                break
            estados = novos
        return estados
