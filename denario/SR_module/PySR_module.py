import numpy as np
from symbolic_regression import SymbolicRegression, SRResult
from typing import List, Optional
from pysr import PySRRegressor
from sklearn.metrics import r2_score

class PySRModule(SymbolicRegression):
    def __init__(self, max_complexity=20, n_iterations=40, populations=15, population_size=33):
        super().__init__(
            n_iterations=n_iterations,
            max_complexity=max_complexity,
            populations=populations,
        )
        
        self.model = PySRRegressor(
            niterations=self.n_iterations,
            maxsize=self.max_complexity,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            populations=self.populations,
            timeout_in_seconds=self.timeout_seconds,
            progress=True,
        )
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> SRResult:
        """
        Run symbolic regression on data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            variable_names: Names for variables (optional)
            
        Returns:
            SRResult with equation and metrics
        """
        
        self.model.fit(X, y, variable_names=variable_names)
        
        # Extract best equation
        best_idx = self.model.equations_["loss"].idxmin()
        best = self.model.equations_.loc[best_idx]
        
        # Predict and compute R2
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return SRResult(
            equation=str(best["equation"]),
            latex=self.model.latex(best_idx),
            sympy=self.model.sympy(best_idx),
            loss=float(best["loss"]),
            complexity=int(best["complexity"]),
            r2=r2,
            pareto_front=self.model.equations_[["equation", "loss", "complexity"]].copy(),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)