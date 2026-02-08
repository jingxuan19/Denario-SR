from attr import dataclass
from git import Optional
import numpy as np
from traitlets import Any
import pandas as pd
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
import os
import glob
import json

@dataclass
class SRResult:
    """Result from symbolic regression."""
    equation: str
    latex: str
    sympy: Any
    loss: float
    complexity: int
    r2: float
    pareto_front: pd.DataFrame
    variable_names: List[str]
    target_name: str
    
@dataclass 
class SRConfig:
    """Configuration for symbolic regression."""
    n_iterations: int = 40
    max_complexity: int = 25
    binary_operators: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    unary_operators: List[str] = field(default_factory=lambda: ["sin", "cos", "exp", "log", "sqrt"])
    populations: int = 15
    timeout_seconds: Optional[int] = 300

class SymbolicRegression:
    """
    Runs symbolic regression on experimental data.
    
    Integrates with Denario's pipeline to discover equations from generated data.
    """
    
    def __init__(self, work_dir: str, SR_module, config: Optional[SRConfig] = None):
        self.work_dir = work_dir
        self.SR_module = SR_module
        self.config = config or SRConfig()
        self.results: List[SRResult] = []
        
    def find_data_files(self, data_dir: str = "data") -> List[str]:
        """Find all CSV files in the data directory."""
        data_path = os.path.join(self.work_dir, data_dir)
        if not os.path.exists(data_path):
            return []
        return glob.glob(os.path.join(data_path, "*.csv"))
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: Optional[List[str]] = None,
        target_name: str = "y"
    ) -> SRResult:
        """
        Fit symbolic regression model to data.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
            variable_names: Optional names for each feature
            target_name: Name of the target variable
            
        Returns:
            SRResult with best equation and metadata
        """        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        
        result = self.SR_module.fit(X, y, variable_names=variable_names)
        
        self.results.append(result)
        return result
    
    def fit_from_csv(
        self, 
        csv_path: str, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None
    ) -> SRResult:
        """
        Fit symbolic regression from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            target_column: Name of the target column
            feature_columns: List of feature column names (default: all except target)
            
        Returns:
            SRResult with best equation and metadata
        """
        df = pd.read_csv(csv_path)
        
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        return self.fit(X, y, variable_names=feature_columns, target_name=target_column)
    
    def auto_discover(self, data_dir: str = "data") -> List[SRResult]:
        """
        Automatically run SR on all CSV files in data directory.
        
        For each CSV, tries to identify target variable and run SR.
        
        Returns:
            List of SRResults for each successful run
        """
        results = []
        csv_files = self.find_data_files(data_dir)
        
        for csv_path in csv_files:
            print(f"[SR] Analyzing {csv_path}...")
            try:
                df = pd.read_csv(csv_path)
                
                # Heuristic: last column is often the target
                # Or look for common target names
                target_candidates = ['y', 'target', 'output', 'result', 'displacement', 'velocity', 'energy']
                
                target_col = None
                for candidate in target_candidates:
                    if candidate in df.columns:
                        target_col = candidate
                        break
                
                if target_col is None:
                    target_col = df.columns[-1]  # Default to last column
                
                feature_cols = [c for c in df.columns if c != target_col]
                
                if len(feature_cols) == 0:
                    print(f"[SR] Skipping {csv_path} - no feature columns")
                    continue
                
                result = self.fit_from_csv(csv_path, target_col, feature_cols)
                results.append(result)
                print(f"[SR] Found equation: {result.equation}")
                
            except Exception as e:
                print(f"[SR] Error processing {csv_path}: {e}")
                continue
        
        return results
    
    def save_results(self, output_dir: str = "sr_results"):
        """Save SR results to files."""
        output_path = os.path.join(self.work_dir, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        for i, result in enumerate(self.results):
            # Save as JSON
            result_dict = {
                "equation": result.equation,
                "latex": result.latex,
                "loss": result.loss,
                "complexity": result.complexity,
                "r2": result.r2,
                "variable_names": result.variable_names,
                "target_name": result.target_name,
            }
            
            with open(os.path.join(output_path, f"sr_result_{i}.json"), 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            # Save Pareto front
            result.pareto_front.to_csv(
                os.path.join(output_path, f"pareto_front_{i}.csv"), 
                index=False
            )
        
        print(f"[SR] Saved {len(self.results)} results to {output_path}")
    
    def get_equations_for_paper(self) -> str:
        """
        Generate a summary of discovered equations for inclusion in paper.
        
        Returns:
            Markdown/LaTeX formatted string with equations
        """
        if not self.results:
            return ""
        
        output = "## Discovered Equations\n\nUsing symbolic regression (PySR), we discovered the following mathematical relationships:\n\n"
        
        for i, result in enumerate(self.results):
            output += f"### Equation {i+1}: {result.target_name}\n\n"
            output += f"$$\n{result.target_name} = {result.latex}\n$$\n\n"
            output += f"- **R² Score**: {result.r2:.4f}\n"
            output += f"- **Complexity**: {result.complexity}\n"
            output += f"- **Variables**: {', '.join(result.variable_names)}\n\n"
        
        return output