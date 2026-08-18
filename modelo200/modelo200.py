import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ==============================================================================
# 1. CONFIGURACOES E HIPERPARAMETROS
# ==============================================================================
@dataclass
class HJMDeepONetConfig:
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"
    tipo_dado: torch.dtype = torch.float32
    
    # Parametros do Modelo HJM / Hull-White
    sigma_const: float = 0.015       # Volatilidade base da taxa forward (1.5% a.a.)
    lambda_reversao: float = 0.10     # Velocidade de decaimento da volatilidade
    
    # Dominio Espaco-Temporal
    T_max: float = 10.0              # Maturidade maxima da curva (10 anos)
    t_max: float = 3.0               # Horizonte de projecao (3 anos)
    num_sensores: int = 30           # Quantidade de vertices sensores f(0, s_m)
    
    # Arquitetura DeepONet
    dimensao_latente: int = 64       # Dimensao p do produto interno
    dimensao_oculta_branch: int = 128
    dimensao_oculta_trunk: int = 128
    num_frequencias_fourier: int = 16
    
    # Treinamento
    tamanho_lote_curvas: int = 64    # Numero de curvas forward por epoca (N_u)
    pontos_consulta_pde: int = 256   # Pontos (t, T) por curva (N_y)
    pontos_condicao_inicial: int = 64
    epocas_adam: int = 1500
    taxa_aprendizado: float = 1e-3

config = HJMDeepONetConfig()

# ==============================================================================
# 2. GERADOR FUNCIONAL DE CURVAS DE JUROS (PARAMETRIZACAO NELSON-SIEGEL)
# ==============================================================================
class GeradorCurvasNelsonSiegel:
    """
    Gera familias de curvas forward continuas f(0, s) baseadas na formulacao
    de Nelson-Siegel com perturbacoes estocasticas de nivel, inclinacao e curvatura.
    """
    def __init__(self, cfg: HJMDeepONetConfig):
        self.cfg = cfg
        self.sensores = torch.linspace(0.01, cfg.T_max, cfg.num_sensores, device=cfg.dispositivo)

    def amostrar_lote_curvas(self, tamanho_lote: int) -> Tuple[torch.Tensor, List]:
        """
        Retorna:
            - matriz_sensores: [tamanho_lote, num_sensores]
            - funcoes_integrais: lista de callables para calculo exato de integral_0^T f(0, s) ds
        """
        # Parametros estocasticos: beta0 (nivel), beta1 (inclinacao), beta2 (curvatura), tau
        beta0 = torch.rand(tamanho_lote, 1, device=self.cfg.dispositivo) * 0.08 + 0.04   # 4% a 12%
        beta1 = (torch.rand(tamanho_lote, 1, device=self.cfg.dispositivo) * 2.0 - 1.0) * 0.05 # -5% a +5%
        beta2 = (torch.rand(tamanho_lote, 1, device=self.cfg.dispositivo) * 2.0 - 1.0) * 0.06 # -6% a +6%
        tau = torch.rand(tamanho_lote, 1, device=self.cfg.dispositivo) * 2.0 + 1.0       # 1 a 3 anos
        
        s = self.sensores.unsqueeze(0) # [1, num_sensores]
        s_tau = s / tau
        fator1 = (1.0 - torch.exp(-s_tau)) / s_tau
        fator2 = fator1 - torch.exp(-s_tau)
        
        # Curva Forward: f(s) = beta0 + beta1 * fator1 + beta2 * fator2
        curvas_sensores = beta0 + beta1 * fator1 + beta2 * fator2
        
        # Adicionar perturbacao browniana suave para generalizacao fora de distribuicao
        ruido = torch.cumsum(torch.randn_like(curvas_sensores) * 0.001, dim=-1)
        curvas_sensores = torch.clamp(curvas_sensores + ruido, min=0.005)
        
        return curvas_sensores, (beta0, beta1, beta2, tau)

    def calcular_integral_exata(self, T: torch.Tensor, params: Tuple) -> torch.Tensor:
        """Calcula integral de 0 a T de f(0, s) ds para a condicao inicial."""
        beta0, beta1, beta2, tau = params
        # Integral analitica do formato Nelson-Siegel:
        # int_0^T f(s) ds = beta0 * T + beta1 * tau * (1 - exp(-T/tau)) + beta2 * [tau * (1 - exp(-T/tau)) - T * exp(-T/tau)]
        T_tau = T / tau
        exp_term = torch.exp(-T_tau)
        termo_b1 = tau * (1.0 - exp_term)
        termo_b2 = termo_b1 - T * exp_term
        return beta0 * T + beta1 * termo_b1 + beta2 * termo_b2

# ==============================================================================
# 3. ARQUITETURA PI-DEEPONET
# ==============================================================================
class IncorporacaoFourier2D(nn.Module):
    """Mapeamento espectral para a Trunk Net capturar derivadas continuas suaves."""
    def __init__(self, num_frequencias: int = 16, escala: float = 1.0):
        super().__init__()
        matriz_projecao = torch.randn(2, num_frequencias) * escala
        self.register_buffer("matriz_projecao", matriz_projecao)

    def forward(self, coordenadas: torch.Tensor) -> torch.Tensor:
        projecao = 2.0 * math.pi * torch.matmul(coordenadas, self.matriz_projecao)
        return torch.cat([torch.sin(projecao), torch.cos(projecao)], dim=-1)

class BranchNet(nn.Module):
    """Processa a curva discretizada f(0, s_m) gerando coeficientes latentes."""
    def __init__(self, dim_entrada: int, dim_oculta: int, dim_saida: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta),
            nn.SiLU(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.SiLU(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.SiLU(),
            nn.Linear(dim_oculta, dim_saida)
        )
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.rede(u)

class TrunkNet(nn.Module):
    """Processa coordenadas continuas de consulta (t, T)."""
    def __init__(self, dim_oculta: int, dim_saida: int, num_frequencias: int):
        super().__init__()
        self.fourier = IncorporacaoFourier2D(num_frequencias=num_frequencias)
        self.rede = nn.Sequential(
            nn.Linear(num_frequencias * 2, dim_oculta),
            nn.Tanh(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.Tanh(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.Tanh(),
            nn.Linear(dim_oculta, dim_saida)
        )
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.fourier(coords)
        return self.rede(h)

class HJM_DeepONet(nn.Module):
    """Modelo DeepONet acoplado para solucao do operador de curvas HJM."""
    def __init__(self, cfg: HJMDeepONetConfig):
        super().__init__()
        self.cfg = cfg
        self.branch = BranchNet(cfg.num_sensores, cfg.dimensao_oculta_branch, cfg.dimensao_latente)
        self.trunk = TrunkNet(cfg.dimensao_oculta_trunk, cfg.dimensao_latente, cfg.num_frequencias_fourier)
        self.bias_global = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensores: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        u_sensores: [N_u, m]
        coords:     [N_u, N_y, 2] -> cada elemento contem (t, T)
        Retorna:
            Preco P(t, T): [N_u, N_y, 1]
        """
        N_u, N_y, _ = coords.shape
        b = self.branch(u_sensores)                # [N_u, p]
        
        # Achatar dimensoes da Trunk Net para avaliacao eficiente
        coords_flat = coords.view(-1, 2)
        t_flat = self.trunk(coords_flat)           # [N_u * N_y, p]
        t = t_flat.view(N_u, N_y, self.cfg.dimensao_latente) # [N_u, N_y, p]
        
        # Produto interno estendido por lote
        b_expandido = b.unsqueeze(1)               # [N_u, 1, p]
        log_P = torch.sum(b_expandido * t, dim=-1, keepdim=True) + self.bias_global # [N_u, N_y, 1]
        
        # Garantir positividade estrita do preco zero-cupom via exponencial
        return torch.exp(log_P)

# ==============================================================================
# 4. OPERADOR RESIDUAL DE FISICA HJM VIA AUTOGRAD
# ==============================================================================
class OperadorFisicoHJM:
    def __init__(self, cfg: HJMDeepONetConfig, modelo: HJM_DeepONet):
        self.cfg = cfg
        self.modelo = modelo

    def calcular_volatilidade_titulo(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Volatilidade integrada sigma_P(t, T) sob decaimento exponencial."""
        tau = torch.clamp(T - t, min=0.0)
        return (self.cfg.sigma_const / self.cfg.lambda_reversao) * (1.0 - torch.exp(-self.cfg.lambda_reversao * tau))

    def calcular_residuos_pde(self, u_sensores: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Calcula o residuo da EDP:
        dP/dt + dP/dT + r(t)*P + 0.5 * sigma_P^2 * P = 0
        """
        coords.requires_grad_(True)
        P = self.modelo(u_sensores, coords) # [N_u, N_y, 1]
        
        # Gradientes de primeira ordem em relacao a (t, T)
        grad_P = torch.autograd.grad(
            P.sum(), coords,
            create_graph=True, retain_graph=True
        )[0] # [N_u, N_y, 2]
        
        P_t = grad_P[:, :, 0:1] # Derivada temporal dP/dt
        P_T = grad_P[:, :, 1:2] # Derivada espacial dP/dT
        
        t = coords[:, :, 0:1]
        T = coords[:, :, 1:2]
        
        # Extrair taxa curta instantanea r_i(t) = - (1/P(t,t)) * dP/dT(t,t)
        # Para coordenadas onde T -> t, aproximamos r(t) localmente
        coords_short_rate = torch.cat([t, t + 1e-4], dim=-1)
        P_short = self.modelo(u_sensores, coords_short_rate)
        grad_short = torch.autograd.grad(
            P_short.sum(), coords_short_rate,
            create_graph=True, retain_graph=True
        )[0]
        P_T_short = grad_short[:, :, 1:2]
        r_t = - P_T_short / torch.clamp(P_short, min=1e-5)
        
        sigma_P = self.calcular_volatilidade_titulo(t, T)
        
        # Residuo HJM exato
        residuo = P_t + P_T + r_t * P + 0.5 * (sigma_P ** 2) * P
        return residuo

# ==============================================================================
# 5. MOTOR DE TREINAMENTO E OTIMIZACAO HIBRIDA
# ==============================================================================
class TreinadorHJMDeepONet:
    def __init__(self, cfg: HJMDeepONetConfig):
        self.cfg = cfg
        self.modelo = HJM_DeepONet(cfg).to(device=cfg.dispositivo, dtype=cfg.tipo_dado)
        self.gerador_curvas = GeradorCurvasNelsonSiegel(cfg)
        self.operador_fisica = OperadorFisicoHJM(cfg, self.modelo)
        self.otimizador_adam = torch.optim.Adam(self.modelo.parameters(), lr=cfg.taxa_aprendizado)
        self.escalonador = torch.optim.lr_scheduler.CosineAnnealingLR(self.otimizador_adam, T_max=cfg.epocas_adam)

    def executar_treinamento(self):
        print(f"=== INICIANDO TREINAMENTO DO PI-DEEPONET (HJM OPERATOR) ===")
        print(f"Dispositivo: {self.cfg.dispositivo.upper()} | Precisao: {self.cfg.tipo_dado}")
        
        for epoca in range(1, self.cfg.epocas_adam + 1):
            self.otimizador_adam.zero_grad()
            
            # 1. Gerar lote de curvas de entrada f(0, s)
            u_sensores, params = self.gerador_curvas.amostrar_lote_curvas(self.cfg.tamanho_lote_curvas)
            
            # 2. Amostrar coordenadas de consulta no interior: 0 <= t <= t_max, t <= T <= T_max
            t_col = torch.rand(self.cfg.tamanho_lote_curvas, self.cfg.pontos_consulta_pde, 1, device=self.cfg.dispositivo) * self.cfg.t_max
            duracao_restante = torch.rand(self.cfg.tamanho_lote_curvas, self.cfg.pontos_consulta_pde, 1, device=self.cfg.dispositivo) * (self.cfg.T_max - t_col)
            T_col = t_col + duracao_restante
            coords_interior = torch.cat([t_col, T_col], dim=-1)
            
            # Perda Física (Resíduo da EDP de HJM)
            res_pde = self.operador_fisica.calcular_residuos_pde(u_sensores, coords_interior)
            perda_pde = torch.mean(res_pde ** 2)
            
            # 3. Perda da Condição Inicial (t = 0): P(0, T) = exp(- \int_0^T f(0, s) ds)
            T_ic = torch.rand(self.cfg.tamanho_lote_curvas, self.cfg.pontos_condicao_inicial, 1, device=self.cfg.dispositivo) * self.cfg.T_max
            t_ic = torch.zeros_like(T_ic)
            coords_ic = torch.cat([t_ic, T_ic], dim=-1)
            
            P_pred_ic = self.modelo(u_sensores, coords_ic)
            integral_exata = self.gerador_curvas.calcular_integral_exata(T_ic, params)
            P_exato_ic = torch.exp(-integral_exata)
            perda_ic = torch.mean((P_pred_ic - P_exato_ic) ** 2)
            
            # 4. Perda no Vencimento (T = t): P(t, t) = 1.0
            t_mat = torch.rand(self.cfg.tamanho_lote_curvas, 32, 1, device=self.cfg.dispositivo) * self.cfg.t_max
            coords_mat = torch.cat([t_mat, t_mat], dim=-1)
            P_mat = self.modelo(u_sensores, coords_mat)
            perda_mat = torch.mean((P_mat - 1.0) ** 2)
            
            # Perda Composta Total
            perda_total = perda_pde + 20.0 * perda_ic + 10.0 * perda_mat
            perda_total.backward()
            self.otimizador_adam.step()
            self.escalonador.step()
            
            if epoca % 300 == 0 or epoca == 1:
                print(f"Epoca {epoca:04d} | Perda Total: {perda_total.item():.6e} | "
                      f"Residuo HJM: {perda_pde.item():.6e} | Erro IC: {perda_ic.item():.6e}")

        print("Treinamento Adam concluido com sucesso.")

# ==============================================================================
# 6. EXTRATOR DE IMUNIZACAO E GREGAS FUNCIONAIS EM TEMPO REAL
# ==============================================================================
class ExtratorImunizacaoHJM:
    """Calcula metricas de risco de taxa de juros instantaneamente via Autograd."""
    @staticmethod
    def avaliar_risco_curva(modelo: HJM_DeepONet, u_curva: torch.Tensor, t_val: float, T_val: float) -> Dict[str, float]:
        coords = torch.tensor([[[t_val, T_val]]], device=u_curva.device, dtype=torch.float32, requires_grad=True)
        u_batch = u_curva.unsqueeze(0) if u_curva.ndim == 1 else u_curva
        
        P = modelo(u_batch, coords)
        
        # Derivadas primeira e segunda em relacao a T
        grad_P = torch.autograd.grad(P.sum(), coords, create_graph=True)[0]
        P_t = grad_P[0, 0, 0].item()
        P_T = grad_P[0, 0, 1]
        
        grad_P_T = torch.autograd.grad(P_T.sum(), coords, create_graph=False)[0]
        P_TT = grad_P_T[0, 0, 1].item()
        
        preco = P.item()
        duracao_modificada = - (P_T.item() / preco)
        convexidade = P_TT / preco
        taxa_forward_implicita = duracao_modificada # f(t, T) = - (1/P) dP/dT
        
        return {
            "Preco_Titulo": preco,
            "Taxa_Forward_Implicita": taxa_forward_implicita,
            "Duracao_Modificada": duracao_modificada,
            "Convexidade": convexidade,
            "Theta_Decaimento": P_t
        }

# ==============================================================================
# 7. PIPELINE DE EXECUCAO E TESTE FORA DA AMOSTRA
# ==============================================================================
if __name__ == "__main__":
    # 1. Instanciar e Treinar o Operador
    treinador = TreinadorHJMDeepONet(config)
    treinador.executar_treinamento()
    
    # 2. Teste Fora da Amostra: Choque de Taxas Spot (Stress Test em Tempo Real)
    print("\n=== TESTE DE CHOQUE INSTANTANEO NA CURVA DE JUROS (SEM RETREINAMENTO) ===")
    gerador_teste = GeradorCurvasNelsonSiegel(config)
    
    # Curva Base Normal (Nivel = 10%)
    sensores = gerador_teste.sensores
    curva_base = 0.10 + 0.02 * torch.exp(-sensores / 2.0)
    
    # Curva sob Choque de Inclinacao/Estresse (+300 bps na ponta curta)
    curva_choque = curva_base + 0.03 * torch.exp(-sensores / 1.0)
    
    maturidades_teste = [1.0, 3.0, 5.0, 10.0]
    
    print("\nResultados para Curva Base vs Curva sob Estresse (t = 0.5 anos):")
    print(f"{'Maturidade (T)':<15} | {'Preco Base':<12} | {'Preco Choque':<12} | {'Duracao Base':<12} | {'Convexidade':<12}")
    print("-" * 75)
    
    for T in maturidades_teste:
        metricas_base = ExtratorImunizacaoHJM.avaliar_risco_curva(treinador.modelo, curva_base, t_val=0.5, T_val=T)
        metricas_choque = ExtratorImunizacaoHJM.avaliar_risco_curva(treinador.modelo, curva_choque, t_val=0.5, T_val=T)
        
        print(f"{T:<15.1f} | R$ {metricas_base['Preco_Titulo']:<9.4f} | R$ {metricas_choque['Preco_Titulo']:<9.4f} | "
              f"{metricas_base['Duracao_Modificada']:<12.4f} | {metricas_base['Convexidade']:<12.4f}")

    print("\n=== MODELO PI-DEEPONET EXECUTADO COM SUCESSO ===")