"""
main.py
Script principal para executar e demonstrar o framework PINN Margem Equatorial.
Autor: Luiz Tiago Wilcke
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dados_reais import resumo_dados, DADOS_ME
from utilitarios import device_disponivel

def menu():
    print("\n" + "=" * 70)
    print("  PINN MARGEM EQUATORIAL BRASILEIRA")
    print("  Autor: Luiz Tiago Wilcke")
    print("  Baseado no Volume 3 - Redes Neurais Informadas pela Física")
    print("=" * 70)
    print(f"Dispositivo: {device_disponivel()}")
    print()
    resumo_dados()
    print()
    print("Módulos disponíveis:")
    print("  01 - Fundamentos Petrofísica (Darcy Radial)")
    print("  02 - Escoamento Vertical Multifásico")
    print("  03 - Formulação PINN")
    print("  04 - Anisotropia e Inversão de Permeabilidade")
    print("  05 - Elevação Artificial (Gas Lift / BCS)")
    print("  06-25 - Estruturas prontas para expansão (física do livro)")
    print()
    print("Para treinar um módulo específico:")
    print("  python -m modulos.modulo_01_fundamentos_petrofisica")
    print("  python -m modulos.modulo_05_elevacao_artificial")
    print()
    print("Ou importe e use as classes diretamente em seus scripts.")
    print("=" * 70)


if __name__ == "__main__":
    menu()
