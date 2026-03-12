
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eata_agent.network import PVNetCtx
from eata_agent import symbolics

def test_treelstm_network():
    print("Testing Tree-LSTM Network Integration...")
    
    # Setup
    device = torch.device("cpu")
    
    # Use the finance grammar
    grammar = symbolics.rule_map['finance']
    num_transplant = 5
    
    # Initialize Context (and Model)
    ctx = PVNetCtx(grammar, num_transplant, device)
    print(f"Model initialized. Vocab size: {len(ctx.grammar_vocab)}")
    
    # Dummy Input
    # seq: [2, seq_len] (row 0: time_idx, row 1: feature data)
    seq_len = 50
    seq = np.zeros((2, seq_len))
    seq[1, :] = np.random.randn(seq_len)
    
    # Dummy State (partial expression)
    # Example: f->A, A->A+A, A->x0
    state_str = "f->A,A->A+A,A->x0"
    
    print(f"Testing forward pass with state: {state_str}")
    
    try:
        # Run policy_value
        policy, value, profit = ctx.policy_value(seq, state_str)
        
        print("Forward pass successful.")
        print(f"Policy shape: {policy.shape}")
        print(f"Value: {value.item():.4f}")
        print(f"Profit: {profit.item():.4f}")
        
        # Check policy dimension matches nA (grammar size + transplants - 2 special tokens?)
        # Network output dim is defined as: len(grammar_vocab) + num_transplant - 2
        # PVNetCtx.grammar_vocab = ['f->A'] + base_grammars + placeholders
        # Expected output dim is usually the number of actions.
        # Check mcts.py usage. MCTS uses len(grammars) as nA.
        # The network output layer: self.dist_out = nn.Linear(..., len(self.grammar_vocab) + num_transplant - 2)
        # This seems specific to original implementation logic.
        
        assert not torch.isnan(policy).any(), "Policy contains NaNs"
        assert not torch.isnan(value).any(), "Value is NaN"
        assert not torch.isnan(profit).any(), "Profit is NaN"
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

    print("\nTesting Batch Forward Pass...")
    try:
        # Create a batch of states
        states = [
            "f->A,A->x0",
            "f->A,A->A+A,A->x0,A->x1", # A+A -> x0 + x1 (BFS order in string usually, MCTS is DFS?)
            # MCTS string is sequence of applied rules. 
            # If DFS: f->A, A->A+A, A->x0, A->x1.
            # Tree reconstruction assumes the order in list matches the traversal order.
            # Our `rules_to_tree` implements DFS reconstruction.
        ]
        
        seqs = [seq, seq] # same seq for simplicity
        
        policy_batch, value_batch, profit_batch = ctx.policy_value_batch(seqs, states)
        
        print(f"Batch Policy shape: {policy_batch.shape}")
        print(f"Batch Value shape: {value_batch.shape}")
        
        assert policy_batch.shape[0] == 2
        assert value_batch.shape[0] == 2
        
    except Exception as e:
        print(f"Batch pass failed: {e}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        test_treelstm_network()
        print("\nALL INTEGRATION TESTS PASSED!")
    except Exception as e:
        print("\nTESTS FAILED")
