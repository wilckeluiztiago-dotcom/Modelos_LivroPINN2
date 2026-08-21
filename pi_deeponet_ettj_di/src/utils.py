import numpy as np
from typing import Optional

def rng(semente: Optional[int] = None):
    return np.random.default_rng(semente)

def dias_uteis_frac(anos: float) -> float:
    """Fração de ano sob convenção B3 de 252 dias úteis."""
    return anos  # já em anos; tau = DU/252
