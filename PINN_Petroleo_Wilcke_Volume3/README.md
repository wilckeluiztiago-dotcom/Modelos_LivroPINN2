# Redes Neurais Informadas pela Física – Volume 3
## Engenharia de Petróleo e Poços

**Autor: Luiz Tiago Wilcke**  
Bacharel em Estatística | Especialista em Deep Learning Científico e Engenharia de Petróleo  
**VOLUME III: COMPUTAÇÃO NEURAL APLICADA À SUBSUPERFÍCIE**

---

### Descrição

Sistema modular completo de **25 módulos** implementando Physics-Informed Neural Networks (PINNs) para modelagem de poços de petróleo, baseado integralmente no livro:

> **Redes Neurais Informadas pela Física**  
> Volume 3: Dinâmica Multifásica, Sistemas de Elevação Artificial e Completações Inteligentes

O modelo analisa **tamanho/geometria do poço**, gera **imagens esquemáticas 2D/3D**, e fornece implementações das principais equações físicas e arquiteturas neurais descritas no livro.

### Estrutura do Pacote

```
pinn_petroleo_wilcke/
├── __init__.py
├── main.py                          # Script principal de demonstração
├── config/
│   └── configuracoes.py             # Parâmetros físicos, PINN e geometria
├── utils/
│   └── utilitarios.py               # Conversões, autograd, logging, análise de tamanho
├── visualizacao/
│   └── imagem_poco.py               # Geração de imagens do poço (vertical, horizontal, inteligente, 3D)
├── modulos/
│   ├── modulo01_fundamentos.py      # Cap. 1 – Darcy, Difusividade, Buckley-Leverett, Peng-Robinson
│   ├── modulo02_escoamento_vertical.py  # Cap. 2 – Gradiente de pressão, Two-Fluid Model
│   ├── modulo03_arquitetura_pinn.py     # Cap. 3 – Arquitetura, perda multiobjetivo, MAP
│   ├── modulo04_anisotropia.py          # Cap. 4 – Tensor K, inversão, B-PINN
│   ├── modulo05_elevacao_artificial.py  # Cap. 5 – Gas Lift, BCS, Gibbs
│   ├── modulo06_poco_inteligente.py     # Cap. 6 – ICVs, ICDs, DAE
│   ├── modulo07_problemas_inversos.py   # Cap. 7 – History Matching, HMC, VI
│   ├── modulo08_nao_newtoniano.py       # Cap. 8 – Herschel-Bulkley, Oldroyd-B
│   ├── modulo09_geomecanica.py          # Cap. 9 – Biot, Mohr-Coulomb, fraturamento
│   ├── modulo10_operadores_neurais.py   # Cap. 10 – DeepONet, FNO, PINO
│   ├── modulo11_severe_slugging.py      # Cap. 11 – Severe Slugging
│   ├── modulo12_xpinn.py                # Cap. 12 – Domain Decomposition
│   ├── modulo13_multifidelidade.py      # Cap. 13 – Multi-fidelity DeepONet
│   ├── modulo14_thmc.py                 # Cap. 14 – Acoplamento THMC
│   ├── modulo15_pirl.py                 # Cap. 15 – Physics-Informed RL
│   ├── modulo16_transfno.py             # Cap. 16 – Transient FNO
│   ├── modulo17_pignn.py                # Cap. 17 – Graph Neural Networks
│   ├── modulo18_presal.py               # Cap. 18 – Aplicações Pré-Sal
│   ├── modulo19_composicional.py        # Cap. 19 – Peng-Robinson HPHT
│   ├── modulo20_fluencia_sal.py         # Cap. 20 – Creep de sal
│   ├── modulo21_wormholes.py            # Cap. 21 – Transporte reativo
│   ├── modulo22_driftflux.py            # Cap. 22 – Drift-Flux
│   ├── modulo23_swelling.py             # Cap. 23 – Swelling de folhelhos
│   ├── modulo24_eletromagnetico.py      # Cap. 24 – Maxwell / corrosão
│   └── modulo25_sismica.py              # Cap. 25 – Ondas VTI / FWI
├── figuras/                         # Imagens geradas
└── resultados/                      # Checkpoints e saídas
```

### Variáveis em Português

Todas as variáveis, classes e funções principais utilizam nomenclatura em português conforme o livro:
- `porosidade`, `permeabilidade`, `viscosidade_oleo`, `raio_poco`, `pressao_inicial`
- `gradiente_gravitacional`, `gradiente_friccional`, `fracao_fluxo_agua`
- `saturacao_gas_aprisionada_land`, `kr_gas_imbibicao_killough`
- etc.

### Como Executar

```bash
cd /home/workdir/artifacts
python -m pinn_petroleo_wilcke.main
```

Ou:

```python
from pinn_petroleo_wilcke.modulos.modulo01_fundamentos import FundamentosReservatorio
from pinn_petroleo_wilcke.visualizacao.imagem_poco import GeradorImagemPoco

fund = FundamentosReservatorio()
print(fund.resumo())

gerador = GeradorImagemPoco()
relatorio = gerador.analisar_tamanho_e_gerar_relatorio()
```

### Requisitos

- Python ≥ 3.8
- PyTorch
- NumPy
- Matplotlib

### Autor

**Luiz Tiago Wilcke**  
Especialista em Deep Learning Científico e Engenharia de Petróleo  
VOLUME III: COMPUTAÇÃO NEURAL APLICADA À SUBSUPERFÍCIE

---

© 2024-2026 Luiz Tiago Wilcke. Material baseado no livro PINN Volume 3.
