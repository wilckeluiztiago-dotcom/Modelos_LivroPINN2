"""
Módulo: Acoplamento PINN 2D com Fonte Estocástica RDF
Descrição: Solução da Equação de Poisson Semicondutora não-linear 2D
           com perfil estocástico de dopantes via PyTorch Autograd.
Autor: Luiz Tiago Wilcke
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. CONSTANTES FÍSICAS E ESCALAS ADIMENSIONAIS
# =====================================================================
Q_ELEM = 1.602176634e-19       # Carga elementar (C)
EPS_0 = 8.8541878128e-12       # Permissividade do vácuo (F/m)
EPS_SI = 11.7 * EPS_0          # Permissividade do Silício (F/m)
K_BOLTZ = 1.380649e-23         # Constante de Boltzmann (J/K)
TEMP = 300.0                   # Temperatura (K)
V_T = (K_BOLTZ * TEMP) / Q_ELEM # Potencial térmico (~0.02585 V)
N_I = 1.5e16                   # Concentração intrínseca Si a 300K (m^-3)

# Escalas de normalização
L_REF = 10e-9                  # 10 nm
N_REF = 1e24                   # 1e18 cm^-3 em m^-3

# Coeficiente adimensional da equação de Poisson
GAMMA = (Q_ELEM * (L_REF**2) * N_REF) / (EPS_SI * V_T)
N_I_TILDE = N_I / N_REF


# =====================================================================
# 2. MOTOR DE DOPAGEM ESTOCÁSTICA COM INTERPOLADOR CONTÍNUO
# =====================================================================
class CampoRDFInterpolado:
    """
    Encapsula o mapa discreto de RDF e provê amostragem contínua no domínio
    [0, 1]x[0, 1] usando interpolação bilinear compatível com Autograd.
    """
    def __init__(self, nd_grade_2d: torch.Tensor, device: str = 'cpu'):
        # nd_grade_2d shape: [Ny, Nx] (adimensionalizado por N_REF)
        self.device = device
        # Formato esperado por grid_sample: [Batch, Canais, H, W]
        self.grade_tensor = nd_grade_2d.unsqueeze(0).unsqueeze(0).to(device)
        self.ny, self.nx = nd_grade_2d.shape

    def avaliar(self, xy_colocacao: torch.Tensor) -> torch.Tensor:
        """
        xy_colocacao: Tensor de shape [N, 2] com coordenadas em [0, 1] x [0, 1].
        Mapeia [0, 1] -> [-1, 1] para a função F.grid_sample.
        """
        # Normalização de coordenadas para o padrão do grid_sample [-1, 1]
        xy_norm = (xy_colocacao * 2.0) - 1.0
        # grid_sample espera [1, 1, N, 2]
        grid = xy_norm.unsqueeze(0).unsqueeze(1)
        
        amostrado = F.grid_sample(
            self.grade_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        return amostrado.view(-1, 1)


# =====================================================================
# 3. REDE NEURAL INFORMADA PELA FÍSICA (PINN 2D)
# =====================================================================
class PINNPoisson2D(nn.Module):
    """Arquitetura MLP totalmente conectada com ativações suaves (tanh/sin)."""
    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        camadas = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            camadas.append(nn.Linear(hidden_dim, hidden_dim))
            camadas.append(nn.Tanh())
        camadas.append(nn.Linear(hidden_dim, 1))
        self.rede = nn.Sequential(*camadas)

        # Inicialização Xavier para estabilidade de gradientes de 2ª ordem
        for m in self.rede:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Predição de psi_tilde(x, y)."""
        return self.rede(xy)


# =====================================================================
# 4. CONDICIONADOR DE PERDAS E RESÍDUO DA EDP (EQUAÇÃO DE POISSON)
# =====================================================================
class SolverPoissonPINN:
    def __init__(self, modelo: PINNPoisson2D, rdf_interpolador: CampoRDFInterpolado, device: str = 'cpu'):
        self.modelo = modelo.to(device)
        self.rdf = rdf_interpolador
        self.device = device

    def residuo_pde(self, xy_interior: torch.Tensor) -> torch.Tensor:
        """
        Calcula o resíduo da EDP: R = ∇²ψ + γ * (Nd*(x,y) - Na - 2*ni*sinh(ψ))
        utilizando Autograd para as derivadas espaciais parciais de segunda ordem.
        """
        xy = xy_interior.clone().requires_grad_(True)
        psi = self.modelo(xy)

        # 1. Gradiente de 1ª ordem: [dψ/dx, dψ/dy]
        grad_psi = torch.autograd.grad(
            outputs=psi,
            inputs=xy,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dpsi_dx = grad_psi[:, 0:1]
        dpsi_dy = grad_psi[:, 1:2]

        # 2. Laplaciano (Derivadas de 2ª ordem): d²ψ/dx² + d²ψ/dy²
        d2psi_dx2 = torch.autograd.grad(
            outputs=dpsi_dx,
            inputs=xy,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        d2psi_dy2 = torch.autograd.grad(
            outputs=dpsi_dy,
            inputs=xy,
            grad_outputs=torch.ones_like(dpsi_dy),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]

        laplaciano = d2psi_dx2 + d2psi_dy2

        # 3. Termo fonte estocástico avaliado nas mesmas coordenadas
        nd_estocastico = self.rdf.avaliar(xy.detach())
        na_aceitador = torch.zeros_like(nd_estocastico) # Canal n-MOS / Dopagem base nula

        # 4. Densidade de carga adimensional: ρ_tilde = Nd - Na - 2*ni*sinh(ψ)
        carga_adimensional = nd_estocastico - na_aceitador - (2.0 * N_I_TILDE * torch.sinh(psi))
        
        # Resíduo adimensional da equação de Poisson
        residuo = laplaciano + (GAMMA * carga_adimensional)
        return residuo

    def calcular_perda_contorno(self, 
                                xy_fonte: torch.Tensor, v_fonte: float,
                                xy_dreno: torch.Tensor, v_dreno: float,
                                xy_porta: torch.Tensor, v_porta: float,
                                xy_substrato: torch.Tensor, v_sub: float) -> torch.Tensor:
        """Calcula a perda de Dirichlet nas quatro fronteiras do transistor."""
        # Tensões normalizadas por V_t
        loss_s = F.mse_loss(self.modelo(xy_fonte), torch.full((len(xy_fonte), 1), v_fonte / V_T, device=self.device))
        loss_d = F.mse_loss(self.modelo(xy_dreno), torch.full((len(xy_dreno), 1), v_dreno / V_T, device=self.device))
        loss_g = F.mse_loss(self.modelo(xy_porta), torch.full((len(xy_porta), 1), v_porta / V_T, device=self.device))
        loss_b = F.mse_loss(self.modelo(xy_substrato), torch.full((len(xy_substrato), 1), v_sub / V_T, device=self.device))
        
        return loss_s + loss_d + loss_g + loss_b

    def treinar(self, 
                n_epocas: int = 1500, 
                n_interior: int = 4096, 
                n_borda: int = 256,
                v_fonte: float = 0.0,
                v_dreno: float = 0.6,
                v_porta: float = 0.8,
                v_sub: float = 0.0,
                lr: float = 1e-3):
        
        otimizador = torch.optim.Adam(self.modelo.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=n_epocas, eta_min=1e-5)

        # Fixação dos nós de contorno [0, 1] x [0, 1]
        lin_borda = torch.linspace(0, 1, n_borda, device=self.device).view(-1, 1)
        xy_fonte = torch.cat([torch.zeros_like(lin_borda), lin_borda], dim=1)      # x = 0 (Fonte)
        xy_dreno = torch.cat([torch.ones_like(lin_borda), lin_borda], dim=1)       # x = 1 (Dreno)
        xy_substrato = torch.cat([lin_borda, torch.zeros_like(lin_borda)], dim=1)  # y = 0 (Substrato)
        xy_porta = torch.cat([lin_borda, torch.ones_like(lin_borda)], dim=1)       # y = 1 (Porta)

        print(f"Iniciando treinamento da PINN 2D com γ = {GAMMA:.4e}...")
        for epoca in range(1, n_epocas + 1):
            otimizador.zero_grad()

            # Amostragem quasi-aleatória contínua no interior do canal
            xy_interior = torch.rand((n_interior, 2), device=self.device)

            # Perdas
            res = self.residuo_pde(xy_interior)
            loss_pde = torch.mean(res**2)
            
            loss_bc = self.calcular_perda_contorno(
                xy_fonte, v_fonte, xy_dreno, v_dreno, xy_porta, v_porta, xy_substrato, v_sub
            )

            # Ponderação de perdas (w_pde=1.0, w_bc=20.0 para forçar contorno rígido)
            perda_total = loss_pde + (20.0 * loss_bc)
            
            perda_total.backward()
            otimizador.step()
            scheduler.step()

            if epoca % 300 == 0 or epoca == 1:
                print(f"Época [{epoca:04d}/{n_epocas}] | Perda Total: {perda_total.item():.5e} | "
                      f"PDE (Poisson): {loss_pde.item():.5e} | BC (Contorno): {loss_bc.item():.5e}")


# =====================================================================
# 5. EXECUÇÃO INTEGRADA (SÍNTESE RDF + RESOLUÇÃO PINN)
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Gerar Grade Espacial de Dopagem Sintética com RDF (Ex: 64x64 nós)
    nx_grid, ny_grid = 64, 64
    # Concentração nominal 1.0 (em unidades N_REF = 10^24 m^-3) com perturbação de Poisson
    nd_nominal_grid = torch.ones((ny_grid, nx_grid), device=device)
    # Flutuação Poisson local simulando átomos discretos
    v_cell = (L_REF / nx_grid) * (L_REF / ny_grid) * 1e-9 # volume 2.5D
    lambda_k = torch.clamp(nd_nominal_grid * N_REF * v_cell, min=1.0)
    contagem_k = torch.poisson(lambda_k)
    nd_rdf_grade = (contagem_k / (v_cell * N_REF)).to(device) # Normalizado por N_REF

    # 2. Inicializar o interpolador contínuo de dopagem
    interpolador_rdf = CampoRDFInterpolado(nd_rdf_grade, device=device)

    # 3. Instanciar a PINN e o Solver
    pinn = PINNPoisson2D(hidden_dim=64, num_layers=4)
    solver = SolverPoissonPINN(pinn, interpolador_rdf, device=device)

    # 4. Executar Treinamento
    solver.treinar(
        n_epocas=1200,
        n_interior=2048,
        n_borda=128,
        v_fonte=0.0,
        v_dreno=0.5,
        v_porta=0.7,
        v_sub=0.0,
        lr=2e-3
    )

    # 5. Avaliação do Perfil de Potencial Resultante no Centro do Canal
    pinn.eval()
    with torch.no_grad():
        x_corte = torch.linspace(0, 1, 100, device=device).view(-1, 1)
        xy_meio_canal = torch.cat([x_corte, torch.full_like(x_corte, 0.5)], dim=1)
        psi_pred_tilde = pinn(xy_meio_canal)
        psi_fisico_volts = (psi_pred_tilde * V_T).cpu().numpy()
        
    print(f"\nPotencial eletrostático no centro do canal (x=L/2, y=W/2): {psi_fisico_volts[50, 0]:.4f} V")
