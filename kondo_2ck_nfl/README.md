# Transporte Quântico Não-Fermi Líquido no Efeito Kondo de Dois Canais (2CK)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Momento local \(S=1/2\) em ilha quântica de silício acoplado **simetricamente** a dois reservatórios (\(J_1=J_2\)). O ponto crítico 2CK exibe:

- Entropia residual de Majorana: \(S_{\mathrm{res}}=\frac12 k_B\ln 2\)
- Condutância não-Fermi líquido: \(G(T)=G_{\max}[1-A\sqrt{T/T_K}]\)

---

## 1. Física

| Quantidade | Valor / forma |
|------------|----------------|
| \(S_{\mathrm{res}}\) | \(\frac12 k_B\ln 2\) |
| \(G(T)\) | \(G_{\max}(1-\beta\sqrt{T/T_K})\) |
| Ponto crítico | \(J_1=J_2\) |
| NFL | \(\sqrt{T}\) (não \(T^2\)) |

---

## 2. Equações

### Hamiltoniano 2CK

$$
\mathcal{H}_{2CK}
=
\sum_{\alpha=1}^{2}
\sum_{k\sigma}
\epsilon_k c_{\alpha k\sigma}^\dagger c_{\alpha k\sigma}
+
J_1\mathbf{S}\cdot\mathbf{s}_1(0)
+
J_2\mathbf{S}\cdot\mathbf{s}_2(0)
+
V_{\mathrm{bias}}(N_1-N_2)
$$

### Dinâmica não-Markoviana da ilha

$$
\partial_t\hat\rho_d
=
-\frac{i}{\hbar}[\mathcal{H}_{\mathrm{dot}},\hat\rho_d]
+
\sum_{\alpha}\int_0^\infty d\tau\,
\mathcal{K}_\alpha(\tau)\,[\cdots]
$$

### Condutância NFL

$$
G(V,T)
=
\frac{e^2}{2h}
\left(
1-\beta_{2CK}
\sqrt{
\frac{\max(e\lvert V\rvert,k_B T)}{k_B T_K}
}
\right)
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\bigl\lVert\partial_t\hat\rho+i[\mathcal{H},\hat\rho]-\mathcal{K}[\hat\rho]\bigr\rVert^2
+
\lambda_G
\bigl\lVert
G_\theta(V,T)
-
G_{\mathrm{2CK}}(V,T)
\bigr\rVert^2
+
\lambda_\rho\lvert\mathrm{Tr}\,\hat\rho-1\rvert^2
$$

---

## 4. Estrutura

```
kondo_2ck_nfl/
├── src/
│   ├── fisica_2ck.py
│   ├── rede_pinn_2ck.py
│   ├── residuo_2ck.py
│   └── treinamento_2ck.py
├── examples/kondo_2ck.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/kondo_2ck.py
```

---

**© 2026 Luiz Tiago Wilcke**
