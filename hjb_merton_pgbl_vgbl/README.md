# Otimização de Portfólios de Fundos de Previdência (PGBL/VGBL) via HJB-Merton

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

EDP de **Hamilton–Jacobi–Bellman** com utilidade **CRRA** intertemporal, sob taxa **CDI** e aportes mensais programados, para gestão de **PGBL/VGBL** (Previc/Susep).

---

## 1. Mercado

| Produto | Características |
|---------|-----------------|
| **PGBL** | Dedutível no IR (até 12% renda); tributação regressiva/progressiva no resgate |
| **VGBL** | Sem dedução; IR só sobre rendimento |
| Alocação | Renda fixa (CDI/NTN) vs multimercados |
| Horizonte | Acumulação + desacumulação na aposentadoria |

---

## 2. Formulação Matemática

### HJB-Merton

$$
\frac{\partial v}{\partial t}
+\sup_{\pi,\,c}
\Biggl\{
\bigl[r+\pi(\mu-r)\bigr]x\,v_x
-c\,v_x
+\frac12\pi^2\sigma^2 x^2 v_{xx}
+\frac{c^{1-\gamma}}{1-\gamma}
\Biggr\}
-\rho v
=0
$$

### Políticas ótimas (Merton)

$$
\pi^* = \frac{\mu-r}{\gamma\sigma^2}
\qquad
c^*(x) \propto x
$$

### Dinâmica da riqueza

$$
dx = \bigl[r+\pi(\mu-r)\bigr]x\,dt - c\,dt + A(t)\,dt + \pi\sigma x\,dW
$$

com \(A(t)\) = aporte mensal programado.

---

## 3. PINN

Resolve \(v_\theta(t,x)\); extrai \(\pi^*\) e \(c^*(x)\) para alocação dinâmica e política de resgate.

---

## 4. Estrutura

```
hjb_merton_pgbl_vgbl/
├── src/
│   ├── merton_crra.py
│   ├── rede_pinn_hjb.py
│   ├── residuo_hjb_merton.py
│   └── treinamento_hjb.py
├── examples/hjb_pgbl.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/hjb_pgbl.py
```

---

**© 2026 Luiz Tiago Wilcke**
