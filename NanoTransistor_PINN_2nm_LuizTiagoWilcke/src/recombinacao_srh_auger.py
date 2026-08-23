"""
Módulo: Motor Físico Unificado de Geração e Recombinação (SRH, Auger, Radiativo, Óptico)
Projeto: NanoTransistor PINN 2nm
Autor: Luiz Tiago Wilcke
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import torch

try:
    from parametros_materiais_si import ParametrosSilicio
except ImportError:
    # Fallback seguro caso seja executado fora do pacote src/
    from src.parametros_materiais_si import ParametrosSilicio


@dataclass
class RelatorioRecombinacao:
    """Relatório estruturado com tensores locais e integrais globais."""
    R_srh: torch.Tensor          # Perfil SRH (m⁻³ s⁻¹)
    R_auger: torch.Tensor        # Perfil Auger (m⁻³ s⁻¹)
    R_radiativa: torch.Tensor    # Perfil Radiativo (m⁻³ s⁻¹)
    G_opt: torch.Tensor          # Perfil de Geração Óptica (m⁻³ s⁻¹)
    R_total: torch.Tensor        # Taxa líquida R_net = R_srh + R_auger + R_rad - G (m⁻³ s⁻¹)
    regimes: List[str]           # Mecanismo dominante em cada ponto
    R_max: float                 # Taxa líquida de pico (m⁻³ s⁻¹)
    R_integrada: float           # Recombinação espacial integrada no volume (s⁻¹)
    I_recomb: float              # Corrente elétrica de recombinação (A)


class Recombinacao:
    """
    Motor completo de recombinação e geração para nanotransistores.
    Totalmente tensorizado em PyTorch com suporte a aceleração por GPU e Autograd.
    """
    def __init__(self, mat: Optional[ParametrosSilicio] = None, device: Optional[str] = None):
        self.mat = mat or ParametrosSilicio()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.validar_unidades()

    def validar_unidades(self) -> None:
        """Garante consistência dimensional estrita no Sistema Internacional (SI)."""
        assert self.mat.C_n_auger < 1e-35, "C_n_auger deve estar em m⁶/s (~2.8e-43 m⁶/s)."
        assert self.mat.C_p_auger < 1e-35, "C_p_auger deve estar em m⁶/s (~9.9e-44 m⁶/s)."
        assert self.mat.B_rad < 1e-15, "B_rad deve estar em m³/s (~1e-21 m³/s)."
        assert self.mat.Nc_300 > 1e20, "Nc_300 deve estar em m⁻³ (~2.86e25 m⁻³)."

    def _tensor(self, x: Union[float, list, torch.Tensor]) -> torch.Tensor:
        """Converte entradas para tensor PyTorch preservando gradientes caso existam."""
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=torch.float64, device=self.device)
        return x.to(device=self.device, dtype=torch.float64)

    def modelo_defeitos_espacial(
        self,
        x: torch.Tensor,
        L_canal: float = 2.0e-9,
        sigma_interface: float = 0.2e-9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula perfis de tempo de vida tau_n(x) e tau_p(x) considerando aumento
        de estados de armadilha nas interfaces de fonte/dreno ou óxido (2 nm).
        """
        x_t = self._tensor(x)
        # Atenuação nas duas extremidades do canal
        fator_esq = torch.exp(- (x_t ** 2) / (2 * (sigma_interface ** 2)))
        fator_dir = torch.exp(- ((x_t - L_canal) ** 2) / (2 * (sigma_interface ** 2)))
        perfil_interface = torch.clamp(fator_esq + fator_dir, 0.0, 1.0)

        degradacao = 1.0 - (1.0 - self.mat.tau_interface_factor) * perfil_interface
        tau_n = self.mat.tau_n_srh * degradacao
        tau_p = self.mat.tau_p_srh * degradacao
        return tau_n, tau_p

    def srh(
        self,
        n: torch.Tensor,
        p: torch.Tensor,
        T: float = 300.0,
        Et: Optional[float] = None,
        tau_n: Optional[torch.Tensor] = None,
        tau_p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Recombinação Shockley-Read-Hall (SRH) com nível de armadilha arbitrário Et (eV).
        R_SRH = (np - ni²) / [tau_p * (n + n1) + tau_n * (p + p1)]
        """
        n_t = self._tensor(n)
        p_t = self._tensor(p)
        ni = self.mat.ni(T)
        Eg = self.mat.bandgap(T)

        # Se Et não for especificado, assume armadilha no meio do bandgap (midgap)
        if Et is None:
            Et = Eg / 2.0

        kB_T_eV = self.mat.kB_eV * T
        n1 = self.mat.Nc(T) * torch.exp(torch.tensor(-(Eg - Et) / kB_T_eV, device=self.device))
        p1 = self.mat.Nv(T) * torch.exp(torch.tensor(-Et / kB_T_eV, device=self.device))

        tn = self._tensor(self.mat.tau_n_srh) if tau_n is None else self._tensor(tau_n)
        tp = self._tensor(self.mat.tau_p_srh) if tau_p is None else self._tensor(tau_p)

        numerador = n_t * p_t - (ni ** 2)
        denominador = tp * (n_t + n1) + tn * (p_t + p1) + 1e-45
        return numerador / denominador

    def auger(
        self,
        n: torch.Tensor,
        p: torch.Tensor,
        T: float = 300.0,
        modelo: str = "classico"
    ) -> torch.Tensor:
        """
        Recombinação Auger:
        - 'classico': R_A = (Cn*n + Cp*p)(np - ni²)
        - 'high_injection': Inclui efeito de blindagem dielétrica (screening) para n, p > 1e25 m⁻³
        """
        n_t = self._tensor(n)
        p_t = self._tensor(p)
        ni = self.mat.ni(T)
        delta_np = n_t * p_t - (ni ** 2)

        if modelo == "classico":
            Cn = self.mat.C_n_auger
            Cp = self.mat.C_p_auger
        elif modelo == "high_injection":
            densidade_total = n_t + p_t
            screening = 1.0 / (1.0 + torch.sqrt(densidade_total / 1e25 + 1e-12))
            Cn = self.mat.C_n_auger * screening
            Cp = self.mat.C_p_auger * screening
        else:
            raise ValueError(f"Modelo Auger não suportado: {modelo}")

        return (Cn * n_t + Cp * p_t) * delta_np

    def radiativa(self, n: torch.Tensor, p: torch.Tensor, T: float = 300.0) -> torch.Tensor:
        """Recombinação radiativa banda a banda: R_rad = B * (np - ni²)."""
        n_t = self._tensor(n)
        p_t = self._tensor(p)
        ni = self.mat.ni(T)
        return self.mat.B_rad * (n_t * p_t - (ni ** 2))

    def geracao_optica(
        self,
        alpha_abs: Union[float, torch.Tensor],
        fluxo_fotons: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Geração óptica G_opt = alpha_abs * Phi (m⁻³ s⁻¹)."""
        return self._tensor(alpha_abs) * self._tensor(fluxo_fotons)

    def diagnostico(
        self,
        r_srh: torch.Tensor,
        r_auger: torch.Tensor,
        r_rad: torch.Tensor,
        g_opt: torch.Tensor
    ) -> List[str]:
        """Classifica o regime físico dominante ponto a ponto."""
        srh_f = torch.abs(r_srh).detach().cpu().flatten()
        aug_f = torch.abs(r_auger).detach().cpu().flatten()
        rad_f = torch.abs(r_rad).detach().cpu().flatten()
        g_f = torch.abs(g_opt).detach().cpu().flatten()

        regimes = []
        for s, a, r, g in zip(srh_f, aug_f, rad_f, g_f):
            valores = {"Equilíbrio": 0.0, "SRH": s.item(), "Auger": a.item(), "Radiativa": r.item(), "Geração Óptica": g.item()}
            dominante = max(valores, key=valores.get)
            regimes.append(f"{dominante} Dominante" if valores[dominante] > 1e-10 else "Equilíbrio Térmico")
        return regimes

    def total(
        self,
        n: torch.Tensor,
        p: torch.Tensor,
        T: float = 300.0,
        x: Optional[torch.Tensor] = None,
        Et: Optional[float] = None,
        tau_n: Optional[torch.Tensor] = None,
        tau_p: Optional[torch.Tensor] = None,
        G_opt: Optional[torch.Tensor] = None,
        modelo_auger: str = "classico",
        area_transversal: float = 4.0e-18  # 2 nm x 2 nm = 4 nm² = 4e-18 m²
    ) -> RelatorioRecombinacao:
        """
        Executa a síntese física completa:
        R_net(x) = R_SRH(x) + R_Auger(x) + R_rad(x) - G_opt(x)
        Calcula a recombinação espacialmente integrada e a corrente de recombinação.
        """
        n_t = self._tensor(n)
        p_t = self._tensor(p)
        g_t = torch.zeros_like(n_t) if G_opt is None else self._tensor(G_opt)

        # Se malha espacial x for passada e tempos não forem fornecidos, calcula defeitos de interface
        if x is not None and tau_n is None and tau_p is None:
            tau_n, tau_p = self.modelo_defeitos_espacial(x)

        # Cálculo das taxas locais
        r_srh = self.srh(n_t, p_t, T=T, Et=Et, tau_n=tau_n, tau_p=tau_p)
        r_auger = self.auger(n_t, p_t, T=T, modelo=modelo_auger)
        r_rad = self.radiativa(n_t, p_t, T=T)

        r_net = r_srh + r_auger + r_rad - g_t

        # Integração espacial para corrente: R_int = Across * integral(R_net(x) dx)
        r_integrada = 0.0
        i_recomb = 0.0
        if x is not None:
            x_t = self._tensor(x)
            if x_t.numel() > 1:
                # Usa integração trapezoidal mantendo coerência
                integral_1d = torch.trapezoid(r_net.detach(), x_t.detach()).item()
                r_integrada = float(integral_1d * area_transversal)
                i_recomb = float(self.mat.q * r_integrada)

        regimes = self.diagnostico(r_srh, r_auger, r_rad, g_t)
        r_max = float(torch.max(torch.abs(r_net)).detach().item())

        return RelatorioRecombinacao(
            R_srh=r_srh,
            R_auger=r_auger,
            R_radiativa=r_rad,
            G_opt=g_t,
            R_total=r_net,
            regimes=regimes,
            R_max=r_max,
            R_integrada=r_integrada,
            I_recomb=i_recomb
        )

    def termo_fonte_continuidade(
        self,
        n: torch.Tensor,
        p: torch.Tensor,
        T: float = 300.0,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retorna diretamente o termo de fonte R_net (m⁻³ s⁻¹) totalmente diferenciável
        para uso no resíduo da Equação de Continuidade ou Loss da PINN:
        Resíduo_n = (1/q) * div(Jn) - R_net
        Resíduo_p = -(1/q) * div(Jp) - R_net
        """
        n_t = self._tensor(n)
        p_t = self._tensor(p)
        tau_n, tau_p = (None, None)
        if x is not None:
            tau_n, tau_p = self.modelo_defeitos_espacial(x)

        r_s = self.srh(n_t, p_t, T=T, tau_n=tau_n, tau_p=tau_p)
        r_a = self.auger(n_t, p_t, T=T)
        r_r = self.radiativa(n_t, p_t, T=T)
        return r_s + r_a + r_r
