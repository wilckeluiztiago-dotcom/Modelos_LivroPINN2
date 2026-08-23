"""
Módulo: Otimização Híbrida Adam → L-BFGS com pesos adaptativos
Autor: Luiz Tiago Wilcke
"""

import torch


def treinar_adam(modelo, perda_fn, x_col, epochs=1500, lr=1e-3,
                 perfil=None, log_every=150, device="cpu"):
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epochs)
    historico = []

    for ep in range(epochs):
        otimizador.zero_grad()
        loss, det = perda_fn(modelo, x_col, perfil=perfil)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 2.0)
        otimizador.step()
        scheduler.step()
        historico.append(loss.item())

        # adaptação simples de pesos (a cada 300 épocas)
        if ep > 0 and ep % 300 == 0:
            if det["bc"] > 10 * det["poisson"]:
                perda_fn.lambda_bc *= 0.8
            elif det["poisson"] > 5 * det["bc"]:
                perda_fn.lambda_p *= 0.9

        if ep % log_every == 0:
            print(f"Adam {ep:5d} | Loss {loss.item():.3e} | "
                  f"P {det['poisson']:.2e} C {det['continuidade']:.2e} BC {det['bc']:.2e}")
    return historico


def treinar_lbfgs(modelo, perda_fn, x_col, max_iter=60, perfil=None):
    otimizador = torch.optim.LBFGS(
        modelo.parameters(), lr=0.8, max_iter=25,
        history_size=60, line_search_fn="strong_wolfe",
        tolerance_grad=1e-8, tolerance_change=1e-10
    )
    historico = []

    def closure():
        otimizador.zero_grad()
        loss, _ = perda_fn(modelo, x_col, perfil=perfil)
        loss.backward()
        return loss

    n_steps = max(1, max_iter // 25)
    for i in range(n_steps):
        loss_val = otimizador.step(closure)
        historico.append(float(loss_val))
        print(f"L-BFGS {i:3d} | Loss {float(loss_val):.3e}")
    return historico
