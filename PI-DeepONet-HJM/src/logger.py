"""
Módulo 18: Logger e Monitoramento
Autor: Luiz Tiago Wilcke
"""

from pathlib import Path
import json
from datetime import datetime


class Logger:
    def __init__(self, diretorio: str = "results/logs"):
        self.diretorio = Path(diretorio)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self.arquivo = self.diretorio / f"treino_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.registros = []

    def registrar(self, epoca: int, perdas: dict):
        entrada = {
            "epoca": epoca,
            "total": float(perdas["total"]),
            "fisica": float(perdas["fisica"]),
            "dados": float(perdas["dados"]),
        }
        self.registros.append(entrada)
        if epoca % 500 == 0:
            self.salvar()

    def salvar(self):
        with open(self.arquivo, "w") as f:
            json.dump(self.registros, f, indent=2)
