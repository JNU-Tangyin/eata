
import time
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eata_agent.model import Model
from eata_agent import symbolics

class MockArgs:
    def __init__(self):
        self.symbolic_lib = 'finance'
        self.max_len = 10 # Keep it short for specific target
        self.max_module_init = 10
        self.num_transplant = 5
        self.num_runs = 1
        self.eta = 0.999
        self.num_aug = 5
        self.exploration_rate = 1.0 / np.sqrt(2)
        self.transplant_step = 100
        self.norm_threshold = 1e-5
        self.device = torch.device('cpu')

def generate_ma_data(n_features=5, n_steps=200, window=5):
    # Generate random features
    X = np.random.randn(n_steps, n_features)
    
    # Target: y = ma(x0, window)
    x0 = X[:, 0]
    
    # MA implementation matching _protected_ma logic
    ma_x0 = symbolics._protected_ma(x0, window)
            
    y = ma_x0
    
    # Minimal noise
    y += 0.001 * np.random.randn(n_steps)
    
    X_tensor = torch.FloatTensor(X).unsqueeze(0) # [1, T, N]
    y_tensor = torch.FloatTensor(y).unsqueeze(0) # [1, T]
    
    return X_tensor, y_tensor

def run_parameter_search_test():
    print("Starting Grammar Parameter Search Test (Target: ma(x0, 5))...")
    
    args = MockArgs()
    # Increase exploration slightly for finding specific parameters
    args.num_transplant = 10
    
    model = Model(args)
    
    print(f"Model initialized. Grammar size: {len(model.base_grammar)}")
    
    X, y = generate_ma_data(n_features=5, n_steps=200, window=5)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    start_time = time.time()
    try:
        all_eqs, all_times, test_scores, records, policy, reward, final_tree = model.run(X, y)
        end_time = time.time()
        
        duration = end_time - start_time
        print("\nTest Completed!")
        print(f"Duration: {duration:.2f} seconds")
        
        best_score = test_scores[0] if test_scores else 0.0
        best_eq = all_eqs[0] if all_eqs else 'N/A'
        
        print(f"Best Score: {best_score}")
        print(f"Best Equation: {best_eq}")
        print(f"Number of MCTS iterations: {len(records) if records else 0}")
        
        # Validation
        if "ma" in best_eq and ("5" in best_eq or "x0" in best_eq):
            print("\nSUCCESS: Found MA structure!")
        else:
            print("\nFAILURE: Did not find MA structure.")
            
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_parameter_search_test()
