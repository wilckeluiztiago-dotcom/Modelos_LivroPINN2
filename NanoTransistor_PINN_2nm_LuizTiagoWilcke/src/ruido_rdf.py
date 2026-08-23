"""
Módulo: NanoTransistor RDF & Statistical Quantum Simulator (2 nm Node)
Autor: Luiz Tiago Wilcke
Descrição: Framework estocástico completo para modelagem atomística de RDF,
           acoplamento Schrödinger-Poisson auto-consistente, recombinação multicanal,
           transporte quântico balístico, sensibilidade global de Sobol/Morris e
           inferência bayesiana inversa via PyTorch.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONSTANTES FÍSICAS FUNDAMENTAIS (SI)
# =============================================================================
Q_ELEM   = 1.602176634e-19        # Carga elementar (C)
H_BAR    = 1.054571817e-34        # Constante de Dirac (J·s)
H_PLANCK = 6.62607015e-34         # Constante de Planck (J·s)
M_0      = 9.1093837015e-31       # Massa do elétron livre (kg)
M_EFF    = 0.19 * M_0             # Massa efetiva transversal no Si (kg)
K_B      = 1.380649e-23           # Constante de Boltzmann (J/K)
EPS_0    = 8.8541878128e-12       # Permissividade do vácuo (F/m)
EPS_SI   = 11.7 * EPS_0           # Permissividade do Silício (F/m)
N_I_300K = 1.5e16                 # Concentração intrínseca a 300K (m^-3)
G_V      = 2                      # Degenerescência de vale para Si (100)


# =============================================================================
# 1. HIERARQUIA DE MODELOS ESTOCÁSTICOS DE DOPAGEM (RDF)
# =============================================================================
class BaseRDF(nn.Module):
    """Classe base abstrata para geradores de RDF."""
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

    def sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class PoissonRDF(BaseRDF):
    """Processo de Poisson pontual não-homogêneo (contagem discreta por volume celular)."""
    def sample(self, Nd_nominal: torch.Tensor, cell_volumes: torch.Tensor, 
               generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna:
            Nd_real: Concentração macroscópica equivalente (m^-3)
            N_atoms: Número inteiro de dopantes em cada célula
        """
        # Taxa de eventos lambda = Nd * Vcell
        lamb = torch.clamp(Nd_nominal * cell_volumes, min=1e-12)
        dist = torch.distributions.Poisson(rate=lamb)
        N_atoms = dist.sample() if generator is None else torch.poisson(lamb, generator=generator)
        Nd_real = N_atoms / cell_volumes
        return Nd_real, N_atoms


class BinomialLatticeRDF(BaseRDF):
    """
    Representação atomística cristalográfica: Ocupação em sítios da rede de Silício.
    Densidade atômica do Silício: ~5.0e28 átomos/m^3 (8 átomos por célula unitária de a=0.543 nm).
    """
    def __init__(self, n_sites: int, cell_volume: float, device: str = 'cpu'):
        super().__init__(device)
        self.n_sites = n_sites
        self.cell_volume = cell_volume
        self.n_si_density = 5.0e28

    def sample(self, Nd_target: torch.Tensor, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p_substituicao = torch.clamp(Nd_target / self.n_si_density, min=1e-9, max=1.0)
        dist = torch.distributions.Binomial(total_count=self.n_sites, probs=p_substituicao)
        k_dopants = dist.sample()
        Nd_real = k_dopants / self.cell_volume
        return Nd_real, k_dopants


class CorrelatedGRFRDF(BaseRDF):
    """Campo Aleatório Gaussiano (GRF) com Kernel de Covariância RBF e decomposição de Fourier (FFT)."""
    def __init__(self, grid_shape: Tuple[int, ...], dx: float, device: str = 'cpu'):
        super().__init__(device)
        self.grid_shape = grid_shape
        self.dims = len(grid_shape)
        self.dx = dx
        self.cell_volume = dx ** self.dims
        
        # Criação da malha de vetores de onda no espaço k
        k_coords = [torch.fft.fftfreq(n, dx, device=device) * 2 * math.pi for n in grid_shape]
        mesh = torch.meshgrid(*k_coords, indexing='ij')
        k_sq = torch.zeros(grid_shape, device=device)
        for ki in mesh:
            k_sq += ki**2
        self.register_buffer('k_sq', k_sq)

    def sample(self, Nd_base: torch.Tensor, correlation_length: torch.Tensor, 
               sigma_flut: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        batch_size = Nd_base.shape[0] if Nd_base.dim() > 0 else 1
        lc = correlation_length.view(-1, *([1] * self.dims))
        var = (sigma_flut**2).view(-1, *([1] * self.dims))

        # Densidade Espectral de Potência RBF
        psd = var * ((2 * math.pi)**(self.dims / 2.0)) * (lc**self.dims) * torch.exp(-0.5 * self.k_sq * (lc**2))
        filtro = torch.sqrt(torch.clamp(psd, min=1e-20))

        # Ruído branco no domínio de Fourier
        ruido_r = torch.randn((batch_size, *self.grid_shape), generator=generator, device=self.device)
        ruido_i = torch.randn((batch_size, *self.grid_shape), generator=generator, device=self.device)
        campo_k = torch.complex(ruido_r, ruido_i) * filtro

        delta = torch.fft.ifftn(campo_k, dim=tuple(range(1, self.dims + 1))).real
        delta = delta / (delta.std(dim=tuple(range(1, self.dims + 1)), keepdim=True) + 1e-12) * sigma_flut.view(-1, *([1]*self.dims))

        # Processo de Cox: Modulação log-normal e discretização de Poisson
        intensidade = Nd_base.view(-1, *([1] * self.dims)) * torch.exp(delta - 0.5 * var)
        taxa = torch.clamp(intensidade * self.cell_volume, min=1e-12)
        n_atoms = torch.poisson(taxa, generator=generator)
        return n_atoms / self.cell_volume


# =============================================================================
# 2. SOLUCIONADOR QUÂNTICO-ELETROSTÁTICO AUTO-CONSISTENTE 1D (2 nm CHANNEL)
# =============================================================================
class SchrodingerPoissonSolver(nn.Module):
    """Solucionador 1D autoconstritor Schrödinger-Poisson com autoestados quânticos e recombinações."""
    def __init__(self, nx: int = 100, l_channel: float = 10e-9, temp: float = 300.0, device: str = 'cpu'):
        super().__init__()
        self.nx = nx
        self.l_channel = l_channel
        self.dx = l_channel / (nx - 1)
        self.temp = temp
        self.vt = (K_B * temp) / Q_ELEM
        self.device = device

        # Operador Laplaciano de diferenças finitas 1D (Matriz Tridiagonal)
        diag = -2.0 * torch.ones(nx, device=device)
        off_diag = torch.ones(nx - 1, device=device)
        self.laplacian_1d = (torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)) / (self.dx**2)
        
        # Operador de Energia Cinética: T = -(hbar^2 / 2m*) d2/dx2
        self.h_kin = -(H_BAR**2 / (2.0 * M_EFF)) * self.laplacian_1d

    def resolver_schrodinger(self, potencial_phi: torch.Tensor, u_confinamento: torch.Tensor, 
                            num_subbands: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve [-hbar^2/2m* d2/dx2 + U_conf(x) - q*phi(x)] psi_n = E_n psi_n
        Retorna: (E_n [num_subbands], psi_n [nx, num_subbands])
        """
        # Potencial efetivo total
        v_eff = u_confinamento - (Q_ELEM * potencial_phi)
        hamiltoniano = self.h_kin + torch.diag(v_eff)

        # Autovalores e autofunções via eigh hermitiana
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltoniano)
        
        # Normalização das funções de onda: integral(|psi|^2 dx) = 1
        psi_n = eigenvectors[:, :num_subbands]
        norm = torch.sqrt(torch.sum(psi_n**2 * self.dx, dim=0, keepdim=True))
        psi_n = psi_n / (norm + 1e-12)
        
        return eigenvalues[:num_subbands], psi_n

    def calcular_densidade_quantica(self, e_n: torch.Tensor, psi_n: torch.Tensor, e_fermi: float) -> torch.Tensor:
        """
        Calcula a densidade eletrônica quântica n(x) considerando subbandas 2D em cada autoestado 1D:
        n_2D(x) = sum_n [ (g_v * m* * kB * T) / (pi * hbar^2) * ln(1 + exp((EF - En)/kBT)) * |psi_n(x)|^2 ]
        """
        coef_2d = (G_V * M_EFF * K_B * self.temp) / (math.pi * (H_BAR**2))
        eta = (e_fermi - e_n) / (K_B * self.temp)
        
        # ln(1 + exp(eta)) computado via F.softplus para evitar overflow
        f_integral = F.softplus(eta)
        populacao_subbandas = coef_2d * f_integral  # [num_subbands]
        
        # Densidade espacial 3D equivalente: sum_n (populacao_n * |psi_n(x)|^2)
        n_x = torch.sum(populacao_subbandas * (psi_n**2), dim=1)
        return n_x

    def resolver_poisson_nao_linear(self, nd_perfil: torch.Tensor, na_perfil: torch.Tensor,
                                    v_gate: float, v_sub: float, phi_init: Optional[torch.Tensor] = None,
                                    max_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
        """Solução eletrostática auto-consistente via Método de Newton-Raphson com amortecimento."""
        phi = torch.zeros(self.nx, device=self.device) if phi_init is None else phi_init.clone()
        phi[0] = v_sub
        phi[-1] = v_gate

        for _ in range(max_iter):
            p = N_I_300K * torch.exp(-phi / self.vt)
            n = N_I_300K * torch.exp(phi / self.vt)
            
            # Densidade de carga e derivada d_rho/d_phi
            rho = Q_ELEM * (p - n + nd_perfil - na_perfil)
            drho_dphi = - (Q_ELEM / self.vt) * (p + n)
            
            # Resíduo da EDP: F(phi) = d2_phi/dx2 + rho/eps
            f_res = (self.laplacian_1d @ phi) + (rho / EPS_SI)
            f_res[0] = phi[0] - v_sub
            f_res[-1] = phi[-1] - v_gate
            
            if torch.max(torch.abs(f_res)) < tol:
                break
                
            # Jacobiano: J = Laplacian + diag(drho/dphi / eps)
            jacobiano = self.laplacian_1d.clone()
            jacobiano += torch.diag(drho_dphi / EPS_SI)
            # Condições de contorno de Dirichlet rígidas
            jacobiano[0, :] = 0.0; jacobiano[0, 0] = 1.0
            jacobiano[-1, :] = 0.0; jacobiano[-1, -1] = 1.0
            
            delta_phi = torch.linalg.solve(jacobiano, -f_res)
            phi += 0.5 * delta_phi # Amortecimento de relaxação

        return phi

    def calcular_recombinacoes(self, n: torch.Tensor, p: torch.Tensor,
                               tau_n: float = 1e-7, tau_p: float = 1e-7,
                               c_n: float = 2.8e-43, c_p: float = 9.9e-44,
                               b_rad: float = 1.0e-21) -> Dict[str, torch.Tensor]:
        """Calcula as taxas de recombinação multicanal: SRH, Auger e Radiativa."""
        np_prod = n * p
        ni_sq = N_I_300K**2
        delta_np = np_prod - ni_sq
        
        # Shockley-Read-Hall (SRH)
        r_srh = delta_np / (tau_p * (n + N_I_300K) + tau_n * (p + N_I_300K) + 1e-25)
        # Auger
        r_auger = (c_n * n + c_p * p) * delta_np
        # Radiativa
        r_rad = b_rad * delta_np
        
        return {'R_SRH': r_srh, 'R_Auger': r_auger, 'R_Rad': r_rad, 'R_Total': r_srh + r_auger + r_rad}

    def calcular_corrente_landauer(self, e_subbands: torch.Tensor, ef_source: float, ef_drain: float) -> torch.Tensor:
        """
        Formalismo de Landauer-Büttiker para nano-canal balístico:
        I_DS = (2*q*kB*T / h) * sum_n [ ln(1 + exp((EF_S - En)/kBT)) - ln(1 + exp((EF_D - En)/kBT)) ]
        """
        f_s = F.softplus((ef_source - e_subbands) / (K_B * self.temp))
        f_d = F.softplus((ef_drain - e_subbands) / (K_B * self.temp))
        
        pre_fator = (2.0 * Q_ELEM * K_B * self.temp) / H_PLANCK
        i_ds = pre_fator * torch.sum(f_s - f_d)
        return torch.clamp(i_ds, min=1e-15)


# =============================================================================
# 3. ANÁLISE DE SENSIBILIDADE GLOBAL (SOBOL + MORRIS + AUTOGRAD)
# =============================================================================
class AnaliseVariabilidade:
    """Ferramentas de Quantificação de Incerteza e Decomposição de Sensibilidade."""
    
    @staticmethod
    def indices_sobol(modelo_fn: Callable[[torch.Tensor], torch.Tensor],
                      limites: List[Tuple[float, float]], n_samples: int = 1024,
                      device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Calcula os índices de primeira ordem (S1) e efeito total (ST) via estimador de Saltelli."""
        d = len(limites)
        low = torch.tensor([lim[0] for lim in limites], device=device)
        high = torch.tensor([lim[1] for lim in limites], device=device)
        
        # Matrizes independentes A e B em [0, 1]^d
        a_u = torch.rand((n_samples, d), device=device)
        b_u = torch.rand((n_samples, d), device=device)
        
        A = low + (high - low) * a_u
        B = low + (high - low) * b_u
        
        y_a = modelo_fn(A)
        y_b = modelo_fn(B)
        var_y = torch.var(torch.cat([y_a, y_b]), unbiased=True)
        
        s1 = torch.zeros(d, device=device)
        st = torch.zeros(d, device=device)
        
        for i in range(d):
            ab_i = A.clone()
            ab_i[:, i] = B[:, i]
            y_ab_i = modelo_fn(ab_i)
            
            s1[i] = torch.mean(y_b * (y_ab_i - y_a)) / (var_y + 1e-15)
            st[i] = (0.5 * torch.mean((y_a - y_ab_i)**2)) / (var_y + 1e-15)
            
        return {'S1': s1, 'ST': st, 'Var_Y': var_y}

    @staticmethod
    def screening_morris(modelo_fn: Callable[[torch.Tensor], torch.Tensor],
                         limites: List[Tuple[float, float]], r_trajetorias: int = 20,
                         niveis: int = 4, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Método dos Efeitos Elementares de Morris para triagem de parâmetros."""
        d = len(limites)
        delta = niveis / (2.0 * (niveis - 1))
        low = torch.tensor([lim[0] for lim in limites], device=device)
        high = torch.tensor([lim[1] for lim in limites], device=device)
        
        efeitos = [[] for _ in range(d)]
        
        for _ in range(r_trajetorias):
            x_u = torch.randint(0, niveis // 2, (d,), device=device).float() / (niveis - 1)
            ordem = torch.randperm(d, device=device)
            
            x_atual = low + (high - low) * x_u
            y_atual = modelo_fn(x_atual.unsqueeze(0)).squeeze(0)
            
            for idx in ordem:
                x_u_novo = x_u.clone()
                x_u_novo[idx] += delta
                x_novo = low + (high - low) * x_u_novo
                y_novo = modelo_fn(x_novo.unsqueeze(0)).squeeze(0)
                
                ee = (y_novo - y_atual) / delta
                efeitos[idx].append(ee.abs())
                x_u = x_u_novo
                y_atual = y_novo
                
        mu_star = torch.tensor([torch.stack(e).mean() for e in efeitos], device=device)
        sigma = torch.tensor([torch.stack(e).std() for e in efeitos], device=device)
        return {'mu_star': mu_star, 'sigma': sigma}

    @staticmethod
    def bootstrap_ic(dados: torch.Tensor, b_reps: int = 2000, alfa: float = 0.05) -> Dict[str, Tuple[float, float, float]]:
        """Calcula estimativas pontuais e intervalos de confiança Bootstrap (95%) com momentos de ordem superior."""
        n = dados.numel()
        indices = torch.randint(0, n, (b_reps, n), device=dados.device)
        boot_samples = dados[indices]
        
        medias = boot_samples.mean(dim=1)
        vars_b = boot_samples.var(dim=1)
        stds = boot_samples.std(dim=1)
        
        z = boot_samples - medias.unsqueeze(1)
        skewness = (z**3).mean(dim=1) / (stds**3 + 1e-15)
        kurtosis = (z**4).mean(dim=1) / (stds**4 + 1e-15) - 3.0
        
        return {
            'media': (dados.mean().item(), torch.quantile(medias, alfa/2).item(), torch.quantile(medias, 1 - alfa/2).item()),
            'variancia': (dados.var().item(), torch.quantile(vars_b, alfa/2).item(), torch.quantile(vars_b, 1 - alfa/2).item()),
            'skewness': (skewness.mean().item(), torch.quantile(skewness, alfa/2).item(), torch.quantile(skewness, 1 - alfa/2).item()),
            'kurtosis': (kurtosis.mean().item(), torch.quantile(kurtosis, alfa/2).item(), torch.quantile(kurtosis, 1 - alfa/2).item()),
        }


# =============================================================================
# 4. INFERÊNCIA INVERSA BAYESIANA VIA MCMC (METROPOLIS-HASTINGS)
# =============================================================================
class InferenciaInversaBayesiana:
    """Inferência a posteriori p(theta | I_obs) para identificar parâmetros físicos de RDF."""
    def __init__(self, forward_fn: Callable[[torch.Tensor], torch.Tensor],
                 prior_mu: torch.Tensor, prior_sigma: torch.Tensor, sigma_ruido_obs: float = 1e-6):
        self.forward = forward_fn
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.sigma_obs = sigma_ruido_obs

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        if torch.any(theta <= 0):
            return torch.tensor(-float('inf'), device=theta.device)
        # Priors Log-Normais
        log_theta = torch.log(theta)
        return -0.5 * torch.sum(((log_theta - self.prior_mu) / self.prior_sigma)**2)

    def log_likelihood(self, theta: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(theta.unsqueeze(0)).squeeze(0)
        return -0.5 * torch.sum(((y_pred - y_obs) / self.sigma_obs)**2)

    def amostrar_mcmc(self, y_obs: torch.Tensor, n_samples: int = 1500,
                      burn_in: int = 300, step_size: float = 0.05) -> torch.Tensor:
        """Cadeia de Markov Monte Carlo (M-H com proposta Gaussiana no espaço logarítmico)."""
        theta_atual = torch.exp(self.prior_mu).clone()
        log_post_atual = self.log_prior(theta_atual) + self.log_likelihood(theta_atual, y_obs)
        
        cadeia = []
        for step in range(n_samples + burn_in):
            # Proposta log-normal
            proposta = theta_atual * torch.exp(torch.randn_like(theta_atual) * step_size)
            log_post_prop = self.log_prior(proposta) + self.log_likelihood(proposta, y_obs)
            
            alpha = log_post_prop - log_post_atual
            if torch.log(torch.rand(1, device=theta_atual.device)) < alpha:
                theta_atual = proposta
                log_post_atual = log_post_prop
                
            if step >= burn_in:
                cadeia.append(theta_atual.clone())
                
        return torch.stack(cadeia)
