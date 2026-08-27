# 📐 Catálogo Completo de Equações

**Autor:** Luiz Tiago Wilcke  
**Obra:** Redes Neurais Informadas pela Física — Volume 3

Todas as equações abaixo são renderizadas automaticamente no GitHub (math support).

---

## Capítulo 1 — Fundamentos da Física de Reservatórios

### Densidade e °API

$$
^\circ\mathrm{API} = \frac{141{,}5}{SG}-131{,}5,\quad SG=\frac{\rho_o}{\rho_w}
$$

### Navier–Stokes → Darcy (média volumétrica)

$$
\rho_f\left(\frac{\partial\mathbf{v}}{\partial t}+\mathbf{v}\cdot\nabla\mathbf{v}\right)=-\nabla P+\mu\nabla^2\mathbf{v}+\rho_f\mathbf{g}
$$

$$
\langle\mathbf{v}\rangle=\mathbf{u}=-\frac{k}{\mu}\bigl(\nabla\langle P\rangle^f-\rho_f\mathbf{g}\bigr)
$$

### Forchheimer

$$
-\frac{\partial P}{\partial x}=\frac{\mu}{k}u+\beta_F\rho u|u|
$$

### Porosidade e Permeabilidade

$$
\phi=\frac{V_v}{V_b}=\frac{V_b-V_s}{V_b}
$$

$$
\phi(P)=\phi_0\exp\bigl[c_f(P-P_0)\bigr]
$$

$$
\mathbf{K}=\begin{pmatrix}k_{xx}&k_{xy}&k_{xz}\\k_{yx}&k_{yy}&k_{yz}\\k_{zx}&k_{zy}&k_{zz}\end{pmatrix}
$$

### Conservação de massa e difusividade

$$
\nabla\cdot(\rho\mathbf{u})+\frac{\partial(\phi\rho)}{\partial t}=q_m
$$

$$
c_l=\frac1\rho\frac{\partial\rho}{\partial P},\quad c_f=\frac1\phi\frac{\partial\phi}{\partial P},\quad c_t=c_l+c_f
$$

$$
\nabla\cdot\left(\frac{\mathbf{K}}{\mu}\nabla P\right)=\phi c_t\frac{\partial P}{\partial t}+\frac{q}{\rho_{\mathrm{std}}}
$$

### Radial estacionário (Dupuit)

$$
\frac1r\frac{\mathrm{d}}{\mathrm{d}r}\left(r\frac{\mathrm{d}P}{\mathrm{d}r}\right)=0
$$

$$
P(r)=P_{wf}+\frac{P_e-P_{wf}}{\ln(r_e/r_w)}\ln\Bigl(\frac{r}{r_w}\Bigr)
$$

$$
Q=\frac{2\pi k h(P_e-P_{wf})}{\mu\ln(r_e/r_w)}
$$

### Buckley–Leverett

$$
k_{ro}(S_w)=\frac{k_o(S_w)}{k},\quad k_{rw}(S_w)=\frac{k_w(S_w)}{k}
$$

$$
f_w(S_w)=\frac{1}{1+\frac{k_{ro}\mu_w}{k_{rw}\mu_o}}
$$

$$
\frac{\partial S_w}{\partial t}+\frac{u_t}{\phi}\frac{\mathrm{d}f_w}{\mathrm{d}S_w}\frac{\partial S_w}{\partial x}=0
$$

### Histerese (Land + Killough)

$$
S_{gt}=\frac{S_{g,hy}}{1+C_{\mathrm{Land}}S_{g,hy}},\quad C_{\mathrm{Land}}=\frac1{S_{gr,\max}}-\frac1{S_{g,\max}^{\mathrm{dr}}}
$$

$$
k_{rg}^{\mathrm{imb}}(S_g)=k_{rg}^{\mathrm{dr}}(S_{g,hy})\left(\frac{S_g-S_{gt}}{S_{g,hy}-S_{gt}}\right)^\alpha
$$

$$
S_{g,\max}^{\mathrm{hist}}(x,t)=\max_{\tau\in[0,t]}S_g(x,\tau)
$$

### Peng–Robinson

$$
P=\frac{RT}{V_m-b}-\frac{a(T)}{V_m(V_m+b)+b(V_m-b)}
$$

$$
a_i=\Omega_a\frac{R^2T_{c,i}^2}{P_{c,i}}\alpha_i,\quad b_i=\Omega_b\frac{RT_{c,i}}{P_{c,i}}
$$

$$
\alpha_i=\bigl[1+m_i(1-\sqrt{T_{r,i}})\bigr]^2,\quad m_i=0.37464+1.54226\omega_i-0.26992\omega_i^2
$$

---

## Capítulo 2 — Escoamento Vertical Multifásico

$$
\frac{\mathrm{d}P}{\mathrm{d}z}=\rho_m g\cos\theta+\frac{f\rho_m u|u|}{2D}+\rho_m u\frac{\mathrm{d}u}{\mathrm{d}z}
$$

$$
\rho_m=H_L\rho_L+(1-H_L)\rho_G
$$

Conservação de massa (Two-Fluid):

$$
\frac{\partial(\alpha_g\rho_g)}{\partial t}+\frac{\partial(\alpha_g\rho_g u_g)}{\partial z}=0
$$

$$
\frac{\partial(\alpha_l\rho_l)}{\partial t}+\frac{\partial(\alpha_l\rho_l u_l)}{\partial z}=0
$$

---

## Capítulo 3 — Formulação PINN

$$
\mathcal{L}(\boldsymbol\theta)=w_d\mathcal{L}_{\mathrm{data}}+w_p\mathcal{L}_{\mathrm{phys}}+w_b\mathcal{L}_{\mathrm{bc}}
$$

$$
\mathcal{L}_{\mathrm{phys}}=\frac1{N_c}\sum_{i=1}^{N_c}\bigl|R(\hat u_{\boldsymbol\theta}(x_i,t_i))\bigr|^2
$$

Equivalência MAP:

$$
\boldsymbol\theta_{\mathrm{MAP}}=\arg\max_{\boldsymbol\theta}\bigl[\log p(\mathrm{dados}|\boldsymbol\theta)+\log p(\boldsymbol\theta)\bigr]
$$

---

## Capítulo 4 — Anisotropia e Inversão

$$
\nabla\cdot(\mathbf{K}\nabla P)=\phi c_t\frac{\partial P}{\partial t}
$$

Regularização Total Variation:

$$
\mathcal{L}_{\mathrm{TV}}=\lambda_{\mathrm{TV}}\int|\nabla k|\,\mathrm{d}V
$$

---

## Capítulo 5 — Elevação Artificial

Equação de onda de Gibbs:

$$
\frac{\partial^2u}{\partial t^2}=c^2\frac{\partial^2u}{\partial x^2}-\alpha\frac{\partial u}{\partial t}
$$

---

## Capítulo 8 — Fluidos Não-Newtonianos

Lei de potência:

$$
\mu_{\mathrm{app}}=K\dot\gamma^{n-1}
$$

Bingham:

$$
\tau=\tau_0+\mu_p\dot\gamma
$$

Herschel–Bulkley:

$$
\tau=\tau_0+K\dot\gamma^n
$$

---

## Capítulo 9 — Geomecânica

Tensão efetiva de Biot:

$$
\boldsymbol\sigma'=\boldsymbol\sigma-\alpha_{\mathrm{Biot}}P\,\mathbf{I}
$$

Mohr–Coulomb:

$$
\mathrm{FS}=\frac{2c\cos\phi+(\sigma_1+\sigma_3)\sin\phi}{\sigma_1-\sigma_3}
$$

Kirsch:

$$
\sigma_{\theta\theta}=(\sigma_H+\sigma_h)-2(\sigma_H-\sigma_h)\cos2\theta-P_w
$$

---

## Capítulos 10–25 — Operadores e Extensões

DeepONet (estrutura):

$$
G_\theta(u)(y)=\sum_{k=1}^p b_k(u)\,t_k(y)
$$

FNO (camada de Fourier):

$$
(K(\phi)v_t)(x)=\mathcal{F}^{-1}\bigl(R_\phi\cdot\mathcal{F}(v_t)\bigr)(x)
$$

Drift-Flux:

$$
u_g=C_0 u_m+u_d
$$

Norton–Bailey (creep de sal):

$$
\dot\varepsilon^{vp}=A\,\sigma_{\mathrm{eq}}^n\exp(-Q/RT)
$$

---

<p align="center">
  <b>Luiz Tiago Wilcke</b><br/>
  Redes Neurais Informadas pela Física — Volume 3
</p>
