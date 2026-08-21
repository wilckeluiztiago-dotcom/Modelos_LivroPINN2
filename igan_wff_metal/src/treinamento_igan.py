"""Treinamento minimax I-GAN."""
import numpy as np
from typing import Dict, List, Optional
from .rede_gan import Gerador, Discriminador
from .perda_igan import perda_discriminador, perda_gerador_total
from .graos_wff import gerar_mapa_graos


def treinar_igan(
    G: Gerador,
    D: Discriminador,
    n_epocas: int = 400,
    batch: int = 8,
    taxa_g: float = 8e-4,
    taxa_d: float = 8e-4,
    lambda_phys: float = 0.4,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    g = np.random.default_rng(semente)
    theta_g = G.parametros_vetor().copy()
    theta_d = D.parametros_vetor().copy()
    n_g, n_d = len(theta_g), len(theta_d)
    hist_g, hist_d, hist_phys = [], [], []
    m_g, m_d = np.zeros_like(theta_g), np.zeros_like(theta_d)
    eps = 1e-5

    for epoca in range(1, n_epocas + 1):
        # batch real
        reals = []
        for b in range(batch):
            _, wf = gerar_mapa_graos(G.nx, G.ny, n_graos=10, semente=int(g.integers(0, 1e9)))
            reals.append(wf)
        wf_real = np.stack(reals, axis=0)
        z = g.normal(size=(batch, G.dim_z))
        wf_fake = G.gerar(z)

        # --- atualiza D ---
        d0 = perda_discriminador(D, wf_real, wf_fake)
        grad_d = np.zeros_like(theta_d)
        idx = g.choice(n_d, size=min(28, n_d), replace=False)
        for j in idx:
            tp = theta_d.copy(); tp[j] += eps
            D.carregar_parametros(tp)
            grad_d[j] = (perda_discriminador(D, wf_real, wf_fake) - d0) / eps
        m_d = 0.9 * m_d + 0.1 * grad_d
        theta_d = theta_d - taxa_d * m_d
        D.carregar_parametros(theta_d)

        # --- atualiza G ---
        z2 = g.normal(size=(batch, G.dim_z))
        g0, adv0, phys0 = perda_gerador_total(G, D, z2, lambda_phys)
        grad_g = np.zeros_like(theta_g)
        idx = g.choice(n_g, size=min(28, n_g), replace=False)
        for j in idx:
            tp = theta_g.copy(); tp[j] += eps
            G.carregar_parametros(tp)
            pj, _, _ = perda_gerador_total(G, D, z2, lambda_phys)
            grad_g[j] = (pj - g0) / eps
        m_g = 0.9 * m_g + 0.1 * grad_g
        theta_g = theta_g - taxa_g * m_g
        G.carregar_parametros(theta_g)

        hist_g.append(g0)
        hist_d.append(d0)
        hist_phys.append(phys0)
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | L_G={g0:.4e} | L_D={d0:.4e} | phys={phys0:.4e}")
        if epoca % 150 == 0:
            taxa_g *= 0.85
            taxa_d *= 0.85

    return {"hist_g": hist_g, "hist_d": hist_d, "hist_phys": hist_phys}
