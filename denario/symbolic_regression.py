from attr import dataclass
from git import Optional
import numpy as np
from traitlets import Any
import pandas as pd
from typing import List

@dataclass
class SRResult:
    """Result from symbolic regression."""
    equation: str
    latex: str
    sympy: Any                  # SymPy expression object
    loss: float                 # MSE loss
    complexity: int             # Equation complexity (tree size)
    r2: float                   # R2 score 
    pareto_front: pd.DataFrame  # All equations on Pareto front

class SymbolicRegression:
    """
    Discovers mathematical equations from data using PySR.
    
    Usage:
        sr = SymbolicRegression()
        result = sr.fit(X, y, variable_names=["mass", "acceleration"])
        print(result.equation)  # "x0 * x1"
        print(result.latex)     # "mass \cdot acceleration"
    """
    
    def __init__(
        self,
        n_iterations: int = 40,
        max_complexity: int = 25,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        populations: int = 15,
        timeout_seconds: Optional[int] = None,
    ):
        self.n_iterations = n_iterations
        self.max_complexity = max_complexity
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["sin", "cos", "exp", "log", "sqrt"]
        self.populations = populations
        self.timeout_seconds = timeout_seconds
        
        self._model = None