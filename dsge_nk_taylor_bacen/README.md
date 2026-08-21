# Modelo DSGE Neo-Keynesiano com Regra de Taylor do Copom/Bacen

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

Sistema de equilíbrio geral com **expectativas racionais** (Curva IS dinâmica, Curva de Phillips e **Regra de Taylor** no estilo Copom/Bacen). A rede neural aprende as **funções de política** sob choques de taxa natural, termos de troca e prêmio fiscal — útil para projeções de juros e inflação em tesouraria.

---

## 1. Mercado / Uso

| Aplicação | Descrição |
|-----------|-----------|
| Projeção Selic / inflação | Cenários para ALM e tesouraria |
| Choques TT | Termos de troca (câmbio / commodities) |
| Prêmio fiscal | Risco soberano embutido na regra |
| IRFs | Resposta a choques de política e demanda |

---

## 2. Formulação Matemática

### Curva IS dinâmica

$$
\hat y_t
=
\mathbb{E}_t[\hat y_{t+1}]
-\frac1\sigma
\bigl(
\hat i_t - \mathbb{E}_t[\hat\pi_{t+1}] - \hat r_t^n
\bigr)
$$

### Curva de Phillips (NK)

$$
\hat\pi_t
=
\beta\,\mathbb{E}_t[\hat\pi_{t+1}]
+\kappa\,\hat y_t
$$

### Regra de Taylor (Copom/Bacen)

$$
\hat i_t
=
\phi_\pi\,\hat\pi_t
+\phi_y\,\hat y_t
+\varepsilon_t^i
$$

com \(\phi_\pi > 1\) (Taylor principle).

### Choques exógenos (AR(1))

$$
\hat r_t^n = \rho_{rn}\hat r_{t-1}^n + \varepsilon_t^{rn},
\quad
\mathrm{TT}_t = \rho_{tt}\mathrm{TT}_{t-1} + \varepsilon_t^{tt},
\quad
\mathrm{fisc}_t = \rho_f\mathrm{fisc}_{t-1} + \varepsilon_t^f
$$

---

## 3. Rede de política

$$
(\hat y_t,\hat\pi_t,\hat i_t)
=
f_\theta\bigl(\hat r_t^n,\,\mathrm{TT}_t,\,\mathrm{fisc}_t\bigr)
$$

Perda = erro em relação à solução RE + resíduo das 3 equações de equilíbrio.

---

## 4. Estrutura

```
dsge_nk_taylor_bacen/
├── src/
│   ├── modelo_nk.py
│   ├── rede_politica.py
│   └── treinamento_politica.py
├── examples/dsge_taylor_bacen.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/dsge_taylor_bacen.py
```

---

**© 2026 Luiz Tiago Wilcke**
