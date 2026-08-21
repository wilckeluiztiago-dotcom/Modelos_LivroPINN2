# Delay-HJB com Espaço Estendido para Inércia de Spin em Spin-FETs 2D

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo:** 37  

EDP de **Bellman não-linear** no domínio estendido \((x,y,t)=(M_t,\,M_{t-\tau},\,t)\), resolvida por **PINN**, para otimização de pulsos **STT** (spin-transfer torque) em canais de **grafeno** ou **WSe₂** com tempo de vida de spin finito \(\tau_{\mathrm{spin}}\) (dinâmica **não-Markoviana**).

---

## 1. Fenômeno Físico

Em Spin-FETs 2D:

1. A polarização de spin \(M_t\) relaxa com tempo de vida \(\tau_{\mathrm{spin}}\).
2. A **inércia de spin** introduz memória: a dinâmica depende de \(M_{t-\tau}\).
3. O controle é o **torque de transferência de spin** \(u_t\) (pulso de corrente).
4. A formulação Markoviana padrão falha; usa-se o **espaço estendido** \((M_t, M_{t-\tau}, t)\).

---

## 2. Equações

### Dinâmica de spin com retardo

$$
dM_t
=
\bigl[-\gamma M_t + \alpha_{\mathrm{STT}} u_t - \beta M_{t-\tau}\bigr]dt
+ \sigma\,dW_t
$$

### Custo e controle

$$
J[u]
=
\mathbb{E}\Biggl[
\int_0^T\Bigl((M_t-M^\star)^2 + \lambda_u u_t^2\Bigr)dt
+(M_T-M^\star)^2
\Biggr]
$$

### Controle ótimo (HJB)

$$
u^\star = -\frac{\alpha_{\mathrm{STT}}}{2\lambda_u}\,V_x
$$

### EDP de Bellman no espaço estendido

$$
-V_t + H\bigl(x,y,V_x,V_{xx}\bigr)=0,
\qquad
x=M_t,\; y=M_{t-\tau}
$$

$$
H
=
\min_u\Bigl\{
f(x,y,u)\,V_x + \tfrac12\sigma^2 V_{xx} + (x-M^\star)^2 + \lambda_u u^2
\Bigr\}
$$

$$
f(x,y,u)=-\gamma x + \alpha_{\mathrm{STT}} u - \beta y
$$

### PINN

$$
\mathcal{J}(\theta)
=
\frac1{N_c}\sum_i\bigl|-V_t+H\bigr|^2_{(x_i,y_i,t_i)}
+\lambda_T\frac1{N_T}\sum_j\bigl|V(x_j,y_j,T)-g\bigr|^2
$$

---

## 3. Estrutura

```
delay_hjb_spin_fet/
├── src/
│   ├── dinamica_spin_retardada.py
│   ├── hjbd_retardado.py
│   ├── rede_pinn_hjb.py
│   └── treinamento_hjb.py
├── examples/delay_hjb_spin_fet.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/delay_hjb_spin_fet.py
```

---

**© 2026 Luiz Tiago Wilcke**
