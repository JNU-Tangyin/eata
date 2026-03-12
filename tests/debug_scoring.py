
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eata_agent import score
from eata_agent import symbolics

def generate_synthetic_data(n_features=5, n_steps=100):
    # Generate random features
    X = np.random.randn(n_steps, n_features)
    
    # Target: y = ma(x0, 10) + x1
    x0 = X[:, 0]
    x1 = X[:, 1]
    
    # MA(x0, 10) implementation matching _protected_ma logic roughly
    # _protected_ma uses cumsum/valid convolution + prefix
    # Let's use the actual function to generate ground truth to ensure match
    ma_x0 = symbolics._protected_ma(x0, 10)
            
    y = ma_x0 + x1
    
    return X, y

def debug_scoring():
    print("Debugging Scoring Logic...")
    
    X, y = generate_synthetic_data(n_features=5, n_steps=100)
    
    # Construct data array for score_with_est: [x0, x1, ..., y]
    # score.py expects [n_features+1, n_steps]
    # But wait, model.py constructs it as:
    # supervision_data = np.vstack([X_transposed, y_reshaped])
    # X_transposed: [n_features, n_steps]
    # y_reshaped: [1, n_steps]
    
    data = np.vstack([X.T, y.reshape(1, -1)])
    print(f"Data shape: {data.shape}")
    
    # Expressions to test
    expressions = [
        "ma(x0, 10)",
        "x1",
        "ma(x0, 10) + x1", # Ground Truth
        "ma(x0, 5) + x1",  # Near Truth
        "0",
        "x0 + x1"
    ]
    
    for eq in expressions:
        print(f"\nEvaluating: {eq}")
        try:
            # score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999)
            r, evaluated_eq = score.score_with_est(eq, 10, data)
            print(f"Score: {r:.6f}")
            print(f"Evaluated Eq: {evaluated_eq}")
            
            # Manually calc MSE to verify
            # Note: score_with_est does constant optimization, so r might be better than raw MSE
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_scoring()
