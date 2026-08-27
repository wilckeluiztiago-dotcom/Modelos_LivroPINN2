# =============================================================================
# Módulo de Visualização e Geração de Imagens do Poço
# Autor: Luiz Tiago Wilcke
# =============================================================================
"""
Gera imagens esquemáticas 2D/3D de poços de petróleo (vertical, horizontal,
direcional, multilateral, inteligente) e plota resultados de PINNs.
Analisa tamanho e geometria visualmente.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Arc, Polygon
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

from ..config.configuracoes import GEOMETRIA, SISTEMA, FISICA

class GeradorImagemPoco:
    """Classe principal para gerar imagens esquemáticas de poços."""

    def __init__(self, geometria: Optional[Dict] = None):
        self.geometria = geometria or {
            "tipo_poco": GEOMETRIA.tipo_poco,
            "profundidade_medida": GEOMETRIA.profundidade_medida,
            "profundidade_vertical": GEOMETRIA.profundidade_vertical,
            "diametro_revestimento": GEOMETRIA.diametro_revestimento,
            "diametro_tubing": GEOMETRIA.diametro_tubing,
            "inclinacao": GEOMETRIA.inclinacao,
            "comprimento_horizontal": GEOMETRIA.comprimento_horizontal,
            "numero_laterais": GEOMETRIA.numero_laterais,
            "tipo_completacao": GEOMETRIA.tipo_completacao,
            "valvulas_icv": GEOMETRIA.valvulas_icv,
            "dispositivos_icd": GEOMETRIA.dispositivos_icd,
        }
        self.figuras_salvas: List[str] = []

    def _configurar_estilo(self):
        """Configura estilo visual profissional."""
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.alpha": 0.3,
        })

    def gerar_esquema_vertical(self, salvar: bool = True, caminho: Optional[str] = None) -> plt.Figure:
        """Gera esquema 2D de poço vertical com completação."""
        self._configurar_estilo()
        fig, ax = plt.subplots(figsize=(6, 12))

        md = self.geometria["profundidade_medida"]
        d_rev = self.geometria["diametro_revestimento"] * 50  # escala visual
        d_tub = self.geometria["diametro_tubing"] * 50

        # Superfície
        ax.axhline(0, color="saddlebrown", linewidth=4, label="Superfície")
        ax.fill_between([-3, 3], 0, -2, color="#8B4513", alpha=0.6)

        # Revestimento
        ax.add_patch(Rectangle((-d_rev/2, -md), d_rev, md,
                               fill=False, edgecolor="#2c3e50", linewidth=3, label="Revestimento"))
        # Tubing
        ax.add_patch(Rectangle((-d_tub/2, -md), d_tub, md,
                               fill=False, edgecolor="#3498db", linewidth=2, linestyle="--", label="Tubing"))

        # Cimentação
        ax.fill_betweenx(np.linspace(-md, 0, 50), -d_rev/2 - 0.3, -d_rev/2, color="#95a5a6", alpha=0.5)
        ax.fill_betweenx(np.linspace(-md, 0, 50), d_rev/2, d_rev/2 + 0.3, color="#95a5a6", alpha=0.5)

        # Reservatório
        ax.axhspan(-md + 50, -md + 50 + FISICA.espessura_reservatorio, color="#f39c12", alpha=0.3, label="Reservatório")

        # Anotações
        ax.annotate(f"MD = {md:.0f} m", xy=(d_rev/2 + 0.5, -md/2), fontsize=9)
        ax.annotate(f"rw = {FISICA.raio_poco:.3f} m", xy=(d_rev/2 + 0.5, -md + 100), fontsize=8)
        ax.annotate(f"h = {FISICA.espessura_reservatorio:.0f} m", xy=(d_rev/2 + 0.5, -md + 80), fontsize=8)

        # Cabeça de poço
        ax.add_patch(FancyBboxPatch((-1.5, 0.2), 3, 1.5, boxstyle="round,pad=0.05",
                                    facecolor="#34495e", edgecolor="black"))
        ax.text(0, 0.95, "Árvore de Natal", ha="center", va="center", color="white", fontsize=8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-md - 50, 3)
        ax.set_xlabel("Raio (escala visual)")
        ax.set_ylabel("Profundidade MD (m)")
        ax.set_title(f"Esquema Poço Vertical\nAutor: Luiz Tiago Wilcke", fontweight="bold")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

        if salvar:
            caminho = caminho or os.path.join(SISTEMA.diretorio_figuras, "poco_vertical.png")
            fig.savefig(caminho, dpi=SISTEMA.dpi_imagem, bbox_inches="tight")
            self.figuras_salvas.append(caminho)
        return fig

    def gerar_esquema_horizontal(self, salvar: bool = True, caminho: Optional[str] = None) -> plt.Figure:
        """Gera esquema de poço horizontal."""
        self._configurar_estilo()
        fig, ax = plt.subplots(figsize=(14, 6))

        tvd = self.geometria["profundidade_vertical"]
        l_horiz = self.geometria["comprimento_horizontal"]
        d_rev = self.geometria["diametro_revestimento"] * 30

        # Vertical section
        ax.plot([0, 0], [0, -tvd], color="#2c3e50", linewidth=4, label="Seção Vertical")
        # Buildup + horizontal
        theta = np.linspace(0, np.pi/2, 50)
        r_build = 150
        x_build = r_build * (1 - np.cos(theta))
        y_build = -tvd + r_build * np.sin(theta)
        ax.plot(x_build, y_build, color="#2c3e50", linewidth=4)
        # Horizontal
        ax.plot([x_build[-1], x_build[-1] + l_horiz], [y_build[-1], y_build[-1]],
                color="#e74c3c", linewidth=4, label="Seção Horizontal")

        # Reservatório
        ax.axhspan(y_build[-1] - 15, y_build[-1] + 15, color="#f39c12", alpha=0.25, label="Reservatório")

        # Anotações
        ax.annotate(f"TVD = {tvd:.0f} m", xy=(-80, -tvd/2), fontsize=9)
        ax.annotate(f"L_horiz = {l_horiz:.0f} m", xy=(x_build[-1] + l_horiz/2, y_build[-1] + 40),
                    ha="center", fontsize=9, color="#c0392b")

        ax.set_xlabel("Deslocamento Horizontal (m)")
        ax.set_ylabel("Profundidade TVD (m)")
        ax.set_title("Esquema Poço Horizontal Direcional\nAutor: Luiz Tiago Wilcke", fontweight="bold")
        ax.invert_yaxis()
        ax.legend(loc="upper right")
        ax.set_aspect("equal", adjustable="datalim")

        if salvar:
            caminho = caminho or os.path.join(SISTEMA.diretorio_figuras, "poco_horizontal.png")
            fig.savefig(caminho, dpi=SISTEMA.dpi_imagem, bbox_inches="tight")
            self.figuras_salvas.append(caminho)
        return fig

    def gerar_esquema_inteligente(self, salvar: bool = True, caminho: Optional[str] = None) -> plt.Figure:
        """Esquema de completação inteligente com ICVs e ICDs."""
        self._configurar_estilo()
        fig, ax = plt.subplots(figsize=(8, 12))

        md = self.geometria["profundidade_medida"]
        n_icv = max(self.geometria["valvulas_icv"], 3)
        n_icd = max(self.geometria["dispositivos_icd"], 5)

        # Poço principal
        ax.add_patch(Rectangle((-0.4, -md), 0.8, md, fill=False, edgecolor="#2c3e50", linewidth=3))

        # Segmentos e ICVs
        posicoes_icv = np.linspace(-md + 100, -200, n_icv)
        for i, y in enumerate(posicoes_icv):
            ax.add_patch(FancyBboxPatch((-0.6, y - 15), 1.2, 30, boxstyle="round,pad=0.02",
                                        facecolor="#e74c3c", edgecolor="black", alpha=0.8))
            ax.text(1.0, y, f"ICV-{i+1}", fontsize=8, va="center", color="#c0392b")

        # ICDs na zona produtora
        posicoes_icd = np.linspace(-md + 50, -md + 50 + FISICA.espessura_reservatorio, n_icd)
        for i, y in enumerate(posicoes_icd):
            ax.plot([-0.5, 0.5], [y, y], color="#27ae60", linewidth=3, solid_capstyle="round")
            ax.text(1.0, y, f"ICD-{i+1}", fontsize=7, va="center", color="#1e8449")

        # Sensores
        ax.plot([0.55]*50, np.linspace(-md, -50, 50), "b--", linewidth=1.5, alpha=0.7, label="Fibra DTS/DAS")

        ax.set_xlim(-2, 3)
        ax.set_ylim(-md - 20, 20)
        ax.set_title("Completação Inteligente com ICVs e ICDs\nAutor: Luiz Tiago Wilcke", fontweight="bold")
        ax.set_ylabel("Profundidade MD (m)")
        ax.invert_yaxis()
        ax.legend(loc="upper right")

        if salvar:
            caminho = caminho or os.path.join(SISTEMA.diretorio_figuras, "poco_inteligente.png")
            fig.savefig(caminho, dpi=SISTEMA.dpi_imagem, bbox_inches="tight")
            self.figuras_salvas.append(caminho)
        return fig

    def gerar_imagem_3d(self, salvar: bool = True, caminho: Optional[str] = None) -> plt.Figure:
        """Gera visualização 3D simplificada do poço no reservatório."""
        self._configurar_estilo()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        tvd = self.geometria["profundidade_vertical"]
        l_h = self.geometria.get("comprimento_horizontal", 800)

        # Trajetória
        z = np.linspace(0, tvd, 100)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        # Buildup
        z_build = np.linspace(tvd - 200, tvd, 40)
        x_build = 200 * (1 - np.cos(np.linspace(0, np.pi/2, 40)))
        y_build = np.zeros_like(z_build)
        # Horizontal
        x_horiz = np.linspace(x_build[-1], x_build[-1] + l_h, 60)
        y_horiz = np.zeros(60)
        z_horiz = np.full(60, tvd)

        ax.plot(x, y, z, "k-", linewidth=3, label="Vertical")
        ax.plot(x_build, y_build, z_build, "b-", linewidth=3, label="Buildup")
        ax.plot(x_horiz, y_horiz, z_horiz, "r-", linewidth=3, label="Horizontal")

        # Bloco do reservatório
        xx, yy = np.meshgrid(np.linspace(-100, x_horiz[-1] + 100, 10),
                             np.linspace(-200, 200, 10))
        zz = np.full_like(xx, tvd)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color="orange")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Profundidade (m)")
        ax.invert_zaxis()
        ax.set_title("Visualização 3D do Poço no Reservatório\nAutor: Luiz Tiago Wilcke")
        ax.legend()

        if salvar:
            caminho = caminho or os.path.join(SISTEMA.diretorio_figuras, "poco_3d.png")
            fig.savefig(caminho, dpi=SISTEMA.dpi_imagem, bbox_inches="tight")
            self.figuras_salvas.append(caminho)
        return fig

    def analisar_tamanho_e_gerar_relatorio(self) -> Dict:
        """Analisa dimensões e gera relatório + imagens."""
        from ..utils.utilitarios import resumo_dimensoes_poco
        resumo = resumo_dimensoes_poco(
            self.geometria["profundidade_medida"],
            self.geometria["diametro_revestimento"],
            self.geometria["diametro_tubing"],
            FISICA.espessura_reservatorio
        )
        # Gerar imagens
        self.gerar_esquema_vertical()
        if self.geometria["tipo_poco"] in ["horizontal", "direcional"]:
            self.gerar_esquema_horizontal()
        if self.geometria["tipo_completacao"] == "intelligent" or self.geometria["valvulas_icv"] > 0:
            self.gerar_esquema_inteligente()
        self.gerar_imagem_3d()

        resumo["imagens_geradas"] = self.figuras_salvas
        resumo["autor"] = "Luiz Tiago Wilcke"
        return resumo

def plotar_perfil_pressao(raio: np.ndarray, pressao: np.ndarray, titulo: str = "Perfil de Pressão Radial",
                          salvar: bool = True) -> plt.Figure:
    """Plota perfil de pressão clássico (Eq. 1.17)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(raio, pressao / 1e6, "b-", linewidth=2)
    ax.set_xlabel("Raio (m)")
    ax.set_ylabel("Pressão (MPa)")
    ax.set_title(f"{titulo}\nAutor: Luiz Tiago Wilcke")
    ax.grid(True, which="both", alpha=0.4)
    if salvar:
        caminho = os.path.join(SISTEMA.diretorio_figuras, "perfil_pressao.png")
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
    return fig

def plotar_saturacao_buckley_leverett(x: np.ndarray, sw: np.ndarray, titulo: str = "Frente Buckley-Leverett",
                                      salvar: bool = True) -> plt.Figure:
    """Plota perfil de saturação de água."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, sw, "b-", linewidth=2, label="Sw")
    ax.set_xlabel("Posição x (m)")
    ax.set_ylabel("Saturação de Água Sw")
    ax.set_ylim(0, 1)
    ax.set_title(f"{titulo}\nAutor: Luiz Tiago Wilcke")
    ax.legend()
    ax.grid(True, alpha=0.4)
    if salvar:
        caminho = os.path.join(SISTEMA.diretorio_figuras, "buckley_leverett.png")
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
    return fig
