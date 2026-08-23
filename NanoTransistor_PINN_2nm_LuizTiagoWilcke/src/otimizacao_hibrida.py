"""
Módulo: Otimização Híbrida Adam → L-BFGS
Autor: Luiz Tiago Wilcke
"""

import torch
from tqdm import tqdm


def treinar_adam(modelo, perda_fn, x_col, epochs=2000, lr=1e-3, device="cpu",
                 perfil=None, log_every=200):
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(otimizador, step_size=500, gamma=0.5)
    historico = []

    for ep in range(epochs):
        otimizador.zero_grad()
        loss, detalhes = perda_fn(modelo, x_col, perfil=perfil)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()
        scheduler.step()
        historico.append(loss.item())
        if ep % log_every == 0:
            print(f"Adam Epoch {ep:5d} | Loss: {loss.item():.3e} | "
                  f"P: {detalhes['poisson']:.2e} C: {detalhes['continuidade']:.2e} BC: {detalhes['bc']:.2e}")
    return historico


def treinar_lbfgs(modelo, perda_fn, x_col, max_iter=50, perfil=None):
    otimizador = torch.optim.LBFGS(
        modelo.parameters(), lr=0.5, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe"
    )

    def closure():
        otimizador.zero_grad()
        loss, _ = perda_fn(modelo, x_col, perfil=perfil)
        loss.backward()
        return loss

    historico = []
    for i in range(max(1, max_iter // 20)):
        loss_val = otimizador.step(closure)
        historico.append(float(loss_val))
        print(f"L-BFGS step {i} | Loss: {float(loss_val):.3e}")
    return historico
