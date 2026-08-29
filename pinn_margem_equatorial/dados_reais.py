"""
dados_reais.py
Parâmetros petrofísicos e volumétricos reais da Margem Equatorial Brasileira.
Fontes: EPE (2024-2025), ANP, Petrobras (poço Morpho FZA-M-59).
Autor: Luiz Tiago Wilcke
"""

import torch

# Dados principais - Foz do Amazonas (porção NW) e Margem Equatorial
DADOS_ME = {
    # Volumes
    "volume_in_place_boe": 17.7e9,          # bilhões de barris de óleo equivalente
    "volume_recuperavel_boe": 6.2e9,
    "fator_recuperacao": 0.35,
    "oleo_recuperavel_bbl": 5.1e9,
    "gas_recuperavel_m3": 167e9,

    # Petrofísica típica de turbiditos cretáceos (Limoeiro, Travosas)
    "porosidade_media": 0.22,               # fração (15-30%)
    "porosidade_min": 0.15,
    "porosidade_max": 0.30,
    "permeabilidade_media_mD": 150.0,       # mD
    "permeabilidade_min_mD": 50.0,
    "permeabilidade_max_mD": 400.0,
    "compressibilidade_poros_1Pa": 1.0e-9,
    "compressibilidade_fluido_1Pa": 1.5e-9,

    # Fluido
    "api_graus": 32.0,                      # óleo leve/médio
    "viscosidade_oleo_cP": 0.8,
    "densidade_oleo_kgm3": 860.0,
    "saturacao_agua_inicial": 0.25,
    "saturacao_oleo_residual": 0.20,

    # Condições de reservatório
    "pressao_inicial_bar": 450.0,
    "temperatura_C": 85.0,
    "espessura_reservatorio_m": 40.0,
    "raio_poco_m": 0.1,
    "raio_drenagem_m": 1500.0,

    # Poço e ambiente (Morpho / FZA-M-59)
    "lamina_agua_m": 2886.0,
    "profundidade_vertical_m": 3500.0,      # estimativa
    "distancia_costa_km": 175.0,

    # Anisotropia típica
    "kv_kh_ratio": 0.1,                     # permeabilidade vertical/horizontal
}

def converter_permeabilidade_mD_para_m2(k_mD: float) -> float:
    """Converte permeabilidade de mD para m²."""
    return k_mD * 9.869233e-16

def obter_tensor(chave: str, device: str = "cpu") -> torch.Tensor:
    """Retorna parâmetro como tensor PyTorch."""
    valor = DADOS_ME[chave]
    return torch.tensor(valor, dtype=torch.float32, device=device)

def resumo_dados():
    print("=" * 60)
    print("DADOS REAIS - MARGEM EQUATORIAL BRASILEIRA")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    print(f"Volume in place (NW Foz): {DADOS_ME['volume_in_place_boe']/1e9:.1f} bilhões boe")
    print(f"Recuperável estimado:     {DADOS_ME['volume_recuperavel_boe']/1e9:.1f} bilhões boe")
    print(f"Porosidade média:         {DADOS_ME['porosidade_media']*100:.0f} %")
    print(f"Permeabilidade média:     {DADOS_ME['permeabilidade_media_mD']:.0f} mD")
    print(f"API médio:                {DADOS_ME['api_graus']:.0f} °")
    print(f"Lâmina d'água (Morpho):   {DADOS_ME['lamina_agua_m']:.0f} m")
    print("=" * 60)

if __name__ == "__main__":
    resumo_dados()
