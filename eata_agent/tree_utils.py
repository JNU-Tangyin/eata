import re
import torch
import numpy as np

class TreeNode:
    def __init__(self, rule_idx, rule_str, hidden_state=None, cell_state=None):
        self.rule_idx = rule_idx
        self.rule_str = rule_str
        self.children = []
        self.parent = None
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.embedding = None # To store rule embedding

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

def get_rhs_non_terminals(rule_str, non_terminals):
    """
    Extracts non-terminal symbols from the RHS of a rule string in order.
    Example: 'A->ite(D,A,A)' with NTs=['A','D'] -> ['D', 'A', 'A']
    """
    if '->' not in rule_str:
        return []
    
    rhs = rule_str.split('->')[1]
    
    # Simple character scanning for single-char NTs
    # This assumes all NTs are single uppercase letters as enforced by our refactoring
    found_nts = []
    for char in rhs:
        if char in non_terminals:
            found_nts.append(char)
            
    return found_nts

def rules_to_tree(rule_list, symbol2idx, non_terminals):
    """
    Reconstructs the derivation tree from a list of rules (pre-order/DFS traversal).
    
    Args:
        rule_list: List of rule strings, e.g., ['f->A', 'A->A+A', ...]
        symbol2idx: Dict mapping rule strings to indices (vocabulary)
        non_terminals: Set/List of non-terminal characters, e.g., ['A', 'W', 'L']
        
    Returns:
        root: TreeNode of the root
    """
    if not rule_list:
        return None

    # Process Root
    root_rule = rule_list[0]
    # Use symbol2idx to get consistent index, default to 0 or handle unknown
    root_idx = symbol2idx.get(root_rule, 0) 
    root = TreeNode(root_idx, root_rule)
    
    rhs_nts = get_rhs_non_terminals(root_rule, non_terminals)
    
    stack = []
    for _ in reversed(rhs_nts):
        stack.append(root)
        
    current_rule_idx = 1
    
    while stack and current_rule_idx < len(rule_list):
        parent = stack.pop()
        
        # Get next rule
        rule_str = rule_list[current_rule_idx]
        rule_idx = symbol2idx.get(rule_str, 0)
        node = TreeNode(rule_idx, rule_str)
        
        # Link to parent
        parent.add_child(node)
        
        # Find new expectations from this rule
        new_nts = get_rhs_non_terminals(rule_str, non_terminals)
        
        # Push new expectations to stack (reverse order for DFS)
        for _ in reversed(new_nts):
            stack.append(node)
            
        current_rule_idx += 1
        
    return root

def batch_trees_to_flat_structure(trees):
    """
    Flattens a batch of trees into a structure suitable for parallel Tree-LSTM computation.
    We group nodes by their height or depth? 
    For Child-Sum Tree-LSTM, we can process from leaves up to root.
    
    Returns:
        node_groups: List of lists. Each inner list contains nodes that can be processed in parallel.
                     (Usually grouped by depth, processing deepest first).
        adjacency: Structure to map children outputs to parents.
    """
    # For PyTorch recursive implementation (slow but simple), we don't need this.
    # For Batched implementation:
    # We need to compute h, c for leaves, then their parents, etc.
    # So we group by "height" (distance to furthest leaf).
    pass 
    # To keep it simple for this experiment, we might use a recursive PyTorch forward function 
    # if the batch size is small. Or use a mask-based approach.
