
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eata_agent import symbolics
from eata_agent import score

def test_finance_grammar_generation():
    print("Testing Finance Grammar Generation...")
    
    # 1. Check Grammar Rules
    n_features = 10
    lookback = 5
    grammar = symbolics.gen_enhanced_finance_grammar(n_features, lookback)
    
    print(f"Grammar size: {len(grammar)}")
    
    # Check for presence of key non-terminals in rules
    has_W = any(r.startswith("W->") for r in grammar)
    has_L = any(r.startswith("L->") for r in grammar)
    has_ma = any("ma(" in r for r in grammar)
    
    assert has_W, "Grammar should contain rules for Window (W->...)"
    assert has_L, "Grammar should contain rules for Lag (L->...)"
    assert has_ma, "Grammar should contain moving average rules"
    
    print("Grammar rule checks passed.")
    
    # 2. Check NTN Map
    ntns = symbolics.ntn_map['finance']
    print(f"Non-terminals: {ntns}")
    assert 'W' in ntns and 'L' in ntns, "ntn_map should contain W and L"
    
    print("NTN map checks passed.")

def test_expression_evaluation():
    print("\nTesting Expression Evaluation...")
    
    # Create dummy data: (n_features, time_steps) + target
    n_features = 2
    time_steps = 100
    data = np.random.randn(n_features + 1, time_steps)
    # Ensure positive values for some features to avoid sqrt/log issues initially
    data[0, :] = np.abs(data[0, :]) + 1.0 
    
    # Define some test expressions manually that use the new parameter structure
    test_eqs = [
        "ma(x0, 10)",
        "delay(x1, 5)",
        "diff(x0, 1)",
        "max_n(x0, 20)",
        "rsi(x0, 14)",
        "volatility(x0, 20)",
        "ma(delay(x0, 5), 10)", # Nested
        "ite(diff(x0, 1) > 0, x1, x0)" # Logic
    ]
    
    for eq in test_eqs:
        try:
            # We use score_with_est to test evaluation
            # tree_size is dummy here
            r, evaluated_eq = score.score_with_est(eq, 10, data)
            print(f"Expression: {eq:30} | Score: {r:.4f} | Result: Success")
            
            # Basic sanity check on score
            if r == 0 and "ite" not in eq: # ite might be 0 if condition is weird, but generally shouldn't be exactly 0 unless error
                 print(f"Warning: Score is 0 for {eq}, possibly failed evaluation.")
                 
        except Exception as e:
            print(f"Expression: {eq:30} | FAILED | Error: {e}")
            raise e

def test_parameter_resolution():
    print("\nTesting Parameter Resolution (W/L)...")
    # This tests if the protected functions handle '10' (string/int) correctly
    # The grammar produces strings like "ma(x0, 10)". 
    # eval() will pass integer 10 to _protected_ma(x1, x2).
    # We need to verify _protected_ma handles x2=10.
    
    x = np.arange(20, dtype=float)
    
    # Test MA
    res_ma = symbolics._protected_ma(x, 10)
    assert len(res_ma) == 20, f"MA result length mismatch: {len(res_ma)}"
    # First 9 values should be expanding mean or similar, 10th value (idx 9) is mean(0..9)
    print("MA test passed.")
    
    # Test Delay
    res_delay = symbolics._protected_delay(x, 5)
    assert len(res_delay) == 20
    assert res_delay[5] == x[0], "Delay logic incorrect"
    print("Delay test passed.")

if __name__ == "__main__":
    try:
        test_finance_grammar_generation()
        test_parameter_resolution()
        test_expression_evaluation()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
