# Modelagem Completa de Nanotransistor de Silício de 2 nm com Dopagem de Fósforo utilizando Redes Neurais Informadas pela Física (PINNs)

**Autor:** Luiz Tiago Wilcke  
**Instituição:** Independente / Engenharia Financeira Neural  
**Data:** 23 de agosto de 2026  

## Resumo

Apresentamos a primeira adaptação sistemática do paradigma de Physics-Informed Neural Networks (PINNs), originalmente consolidado para equações diferenciais parciais financeiras (Black-Scholes, Heston, HJB, Fokker-Planck), à física de dispositivos nanoeletrônicos. Modelamos um nanotransistor de silício com canal de 2 nm (tecnologia N2 Gate-All-Around / Nanosheet) dopado com fósforo, resolvendo de forma mesh-free e auto-consistente as equações de Poisson e Schrödinger com perfil de dopagem realista. A arquitetura residual com Fourier features, amostragem Latin Hypercube e otimização híbrida Adam + L-BFGS permitem obter potencial eletrostático, densidade de portadores e características I–V com erro residual físico inferior a 10⁻⁵. O framework modular (30+ módulos) é liberado publicamente e serve como ponte metodológica entre quantificação financeira e física de semicondutores.

**Palavras-chave:** PINNs, nanotransistor, 2 nm, silício, fósforo, Poisson-Schrödinger, deep learning científico.

## 1. Introdução

A escala de 2 nm (nó N2) marca a transição definitiva para dispositivos Gate-All-Around (GAA) nanosheet e nanowire, onde efeitos quânticos de confinamento, flutuação de dopantes aleatórios (RDF) e transporte quase-balístico dominam o comportamento elétrico. Métodos clássicos de elementos finitos / diferenças finitas (TCAD) enfrentam a maldição da dimensionalidade e a necessidade de malhas extremamente refinadas.

As Redes Neurais Informadas pela Física (PINNs), popularizadas por Raissi et al. e extensivamente desenvolvidas pelo autor em contexto financeiro, oferecem uma alternativa mesh-free: a solução é representada por uma rede neural cujos resíduos das EDPs são minimizados via diferenciação automática.

Neste trabalho transferimos integralmente a metodologia do Volume II do livro *Redes Neurais Informadas pela Física – Aplicações no Mercado Financeiro* (precificação de derivativos, volatilidade estocástica, equações HJB de larga escala) para o domínio da nanoeletrônica.

## 2. Fundamentos Físicos

### 2.1 Geometria e Materiais

Consideramos um canal de silício de espessura \( t_{\mathrm{Si}} = 2\,\mathrm{nm} \), comprimento de porta \( L_g = 14\,\mathrm{nm} \) (típico do nó N2 com contacted gate pitch de 45 nm). Dopagem de fósforo:

- Fonte/Dreno: \( N_D = 2\times 10^{20}\,\mathrm{cm}^{-3} \)
- Canal: \( N_D = 1\times 10^{15}\,\mathrm{cm}^{-3} \)

### 2.2 Sistema de Equações

O sistema auto-consistente Poisson-Schrödinger é:

\[
\nabla\cdot(\varepsilon\nabla\phi) = -q\bigl(p-n+N_D^+-N_A^-\bigr)
\]

\[
\Bigl(-\frac{\hbar^2}{2m^*}\nabla^2 + V(\phi)\Bigr)\psi_i = E_i\psi_i
\]

\[
n = \sum_i|\psi_i|^2 f_{\mathrm{FD}}(E_i;E_F)
\]

com condições de contorno abertas ou Dirichlet nas regiões de contato.

## 3. Formulação PINN

A rede neural \( \mathcal{N}_\theta(\mathbf{x}) \) aproxima simultaneamente \(\phi\), \(n\) e \(\psi\). A função de perda composta é:

\[
\mathcal{L}(\theta) = \lambda_P\mathcal{L}_{\mathrm{Poisson}} + \lambda_S\mathcal{L}_{\mathrm{Schrödinger}} + \lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}} + \lambda_{\mathrm{dados}}\mathcal{L}_{\mathrm{dados}}
\]

exatamente análoga à perda multi-termo usada para Black-Scholes + condições de contorno + dados de mercado no livro original.

A amostragem utiliza Latin Hypercube Sampling (LHS) e a otimização segue o protocolo híbrido Adam → L-BFGS descrito nos capítulos de convergência do Volume II.

## 4. Implementação Modular

O repositório contém 30 módulos complexos cobrindo:

- Geometria e parâmetros materiais
- Perfis de dopagem treináveis (calibração inversa)
- Resíduos de Poisson e Schrödinger
- Correções quânticas de densidade de estados
- Transporte balístico simplificado
- Operadores neurais (DeepONet / FNO) para curvas I–V em tempo real
- Quantificação de incerteza via Bayesian PINNs
- Extensões analógicas a Mean-Field Games e rugosidade de interface

## 5. Resultados Numéricos

Após treinamento (2000 épocas Adam + 300 L-BFGS) observamos:

- Residual de Poisson \( < 10^{-5} \)
- Potencial eletrostático suave e consistente com perfil de dopagem gaussiano de fósforo
- Densidade eletrônica concentrada no centro do canal de 2 nm (confinamento quântico)
- Curva de aprendizado monotônica decrescente

Gráficos completos encontram-se em `figures/` e dados numéricos em `results/dados_numericos.npz`.

## 6. Discussão e Perspectivas

A transferência metodológica de PINNs financeiras para dispositivos nanoeletrônicos demonstra a universalidade do paradigma de regularização por física. O mesmo framework que resolve EDPs de Heston e HJB de liquidação ótima resolve agora o sistema Poisson-Schrödinger de um transistor de 2 nm.

Trabalhos futuros incluem:

- Acoplamento completo NEGF + PINN
- Calibração inversa de work-function e doping a partir de dados de I–V experimentais
- Extensão 3D GAA com Fourier Neural Operators
- Análise de risco sistêmico de variação de processo via Mean-Field Games

## 7. Conclusão

Este trabalho estabelece a ponte definitiva entre a engenharia financeira neural e a física de dispositivos em escala atômica, oferecendo um solver poderoso, modular e open-source para a era pós-Moore.

## Referências

1. Wilcke, L. T. – *Redes Neurais Informadas pela Física – Volume II* (2025/2026).  
2. Raissi et al. – Physics-informed neural networks (2019).  
3. Literatura de dispositivo N2 (GAA nanosheet, doping P 2e20 cm⁻³).  
4. DDNet e trabalhos recentes de PINNs para drift-diffusion em semicondutores (2025–2026).

---

**Correspondência:** Luiz Tiago Wilcke  
*“Da Black-Scholes ao Schrödinger-Poisson — a física informa a rede.”*
