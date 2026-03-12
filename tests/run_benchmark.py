
import time
import torch
import numpy as np
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eata_agent.model import Model

class MockArgs:
    def __init__(self):
        self.symbolic_lib = 'finance'
        self.max_len = 20
        self.max_module_init = 10
        self.num_transplant = 3
        self.num_runs = 1
        self.eta = 0.999
        self.num_aug = 5
        self.exploration_rate = 1.0 / np.sqrt(2)
        self.transplant_step = 100
        self.norm_threshold = 1e-5
        self.device = torch.device('cpu')
        # self.n_vars = 5 # Not used for finance lib?

def generate_synthetic_data(n_features=5, n_steps=100):
    # Generate random features
    X = np.random.randn(n_steps, n_features)
    
    # Target: y = ma(x0, 10) + x1
    # Implement ground truth logic manually
    x0 = X[:, 0]
    x1 = X[:, 1]
    
    # MA(x0, 10)
    window = 10
    ma_x0 = np.zeros_like(x0)
    # Simple moving average
    for i in range(len(x0)):
        if i < window - 1:
            ma_x0[i] = np.mean(x0[:i+1])
        else:
            ma_x0[i] = np.mean(x0[i-window+1:i+1])
            
    y = ma_x0 + x1
    
    # Add small noise
    y += 0.01 * np.random.randn(n_steps)
    
    X_tensor = torch.FloatTensor(X).unsqueeze(0) # [1, T, N]
    y_tensor = torch.FloatTensor(y).unsqueeze(0) # [1, T]
    
    return X_tensor, y_tensor

def run_benchmark():
    print("Starting Benchmark on 'alphacfg-experiment' branch...")
    
    args = MockArgs()
    # Increase resources for a real search
    args.num_transplant = 5 # 5 iterations
    args.num_runs = 1
    args.max_len = 20
    
    model = Model(args)
    
    print(f"Model initialized. Grammar size: {len(model.base_grammar)}")
    print(f"Device: {args.device}")
    
    X, y = generate_synthetic_data(n_features=5, n_steps=200) # More steps for stable stats
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    start_time = time.time()
    try:
        # run(self, X, y=None, previous_best_tree=None)
        all_eqs, all_times, test_scores, records, policy, reward, final_tree = model.run(X, y)
        end_time = time.time()
        
        duration = end_time - start_time
        print("\nBenchmark Completed!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Best Score: {test_scores[0] if test_scores else 'N/A'}")
        print(f"Best Equation: {all_eqs[0] if all_eqs else 'N/A'}")
        print(f"Number of MCTS iterations: {len(records) if records else 0}")
        
    except Exception as e:
        print(f"\nBenchmark Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
