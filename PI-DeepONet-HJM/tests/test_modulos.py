"""Testes básicos dos módulos."""
import sys
sys.path.insert(0, ".")
import torch
from src import CONFIG, PIDeepONetHJM, PipelineHJM
from src.geracao_curvas import gerar_ensemble_curvas, curva_forward_nelson_siegel
from src.matematica_hjm import volatilidade_hjm, drift_livre_arbitragem


def test_config():
    assert CONFIG.num_sensores == 50
    assert CONFIG.dim_latent == 64


def test_modelo_forward():
    modelo = PIDeepONetHJM()
    u, _ = gerar_ensemble_curvas(num_curvas=2)
    t = torch.rand(4, 1)
    T = t + torch.rand(4, 1) * 2
    u_b = u[0:1].expand(4, -1)
    P = modelo(u_b, t, T)
    assert P.shape == (4, 1)
    assert (P > 0).all()


def test_volatilidade():
    t = torch.tensor([[0.5]])
    T = torch.tensor([[2.0]])
    sig = volatilidade_hjm(t, T)
    assert sig.shape == t.shape


if __name__ == "__main__":
    test_config()
    test_modelo_forward()
    test_volatilidade()
    print("Todos os testes passaram.")
