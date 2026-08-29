"""
Módulo 03 - Formulação de Redes Neurais Informadas pela Física (PINNs) em Engenharia de Poços
Arquitetura, perda multiobjetivo, MAP e dilema viés-variância.
Autor: Luiz Tiago Wilcke
~200 linhas
"""

import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utilitarios import RedePINN, residual_mse, PerdaMultiobjetivo, set_seed, device_disponivel
from dados_reais import DADOS_ME

set_seed(42)
DEVICE = device_disponivel()


class FormacaoPINNBase(nn.Module):
    """
    Classe base que implementa a formulação clássica de PINN:
    L = λ_data * L_data + λ_phys * L_phys + λ_bc * L_bc
    Equivalência com estimativa MAP (Maximum A Posteriori).
    """
    def __init__(self, dim_entrada=2, dim_saida=1, camadas_ocultas=[64, 64, 64]):
        super().__init__()
        camadas = [dim_entrada] + camadas_ocultas + [dim_saida]
        self.rede = RedePINN(camadas).to(DEVICE)
        self.perda_mgr = PerdaMultiobjetivo({
            "dados": 10.0,
            "fisica": 1.0,
            "contorno": 10.0
        })

    def forward(self, x):
        return self.rede(x)

    def perda_dados(self, x_data, y_data):
        y_pred = self.forward(x_data)
        return residual_mse(y_pred - y_data)

    def perda_fisica(self, x_col):
        """Deve ser sobrescrito pela física específica."""
        raise NotImplementedError

    def perda_contorno(self, x_bc, y_bc):
        y_pred = self.forward(x_bc)
        return residual_mse(y_pred - y_bc)

    def perda_total(self, x_data, y_data, x_col, x_bc, y_bc):
        L_d = self.perda_dados(x_data, y_data)
        L_p = self.perda_fisica(x_col)
        L_c = self.perda_contorno(x_bc, y_bc)
        perdas = {"dados": L_d, "fisica": L_p, "contorno": L_c}
        return self.perda_mgr(perdas), {k: v.item() for k, v in perdas.items()}


class PINNDifusividadeExemplo(FormacaoPINNBase):
    """Exemplo concreto: equação de difusividade 1D."""
    def __init__(self):
        super().__init__(dim_entrada=2, dim_saida=1)
        self.alpha = torch.tensor(0.1, device=DEVICE)  # difusividade térmica/pressão

    def perda_fisica(self, x_col):
        x_col = x_col.clone().requires_grad_(True)
        u = self.forward(x_col)
        # u_t - alpha * u_xx = 0
        grads = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        u_t = grads[:, 1:2]
        u_x = grads[:, 0:1]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0:1]
        res = u_t - self.alpha * u_xx
        return residual_mse(res)


def demonstrar_formulacao():
    print("=" * 60)
    print("Módulo 03 - Formulação PINN (Autor: Luiz Tiago Wilcke)")
    print("=" * 60)
    print("Paradigma: L_total = λ_d L_data + λ_p L_phys + λ_bc L_bc")
    print("Equivalência estatística: Estimativa de Máxima A Posteriori (MAP)")
    print("Dilema viés-variância controlado pelos pesos λ e pela capacidade da rede.")
    print()
    modelo = PINNDifusividadeExemplo()
    print(f"Rede criada com {sum(p.numel() for p in modelo.parameters())} parâmetros.")
    print("Formulação pronta para acoplamento com física de reservatório equatorial.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrar_formulacao()
