# Modelo Matemático — PINN 3D Poisson GAAFET

**Autor:** Luiz Tiago Wilcke

## Equação de Poisson

$$
\nabla\cdot(\varepsilon\nabla\phi)=-\rho
$$

## Geometria GAAFET

Canal cilíndrico $r\le R_c$, óxido $R_c<r\le R_{\mathrm{ox}}$, gate em $r=R_{\mathrm{ox}}$.

## PINN (Caps. 2–3 do livro)

$$
\mathcal{J}(\theta)=\|\nabla\cdot(\varepsilon\nabla\phi_\theta)+\rho\|^2_{L^2}+\lambda\|\phi_\theta-g\|^2_{\partial}
$$

Amostragem LHS; sem malha espacial.
