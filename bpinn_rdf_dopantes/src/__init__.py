"""
B-PINNs para Incerteza Epistêmica em RDF (Apêndice C.3)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rdf_dopantes import CanalRDF
from .ensemble_bpinn import EnsembleBPINN, MembroPINN
from .rede_bpinn import RedeBayesiana

__all__ = ["CanalRDF", "EnsembleBPINN", "MembroPINN", "RedeBayesiana"]
