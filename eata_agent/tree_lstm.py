import torch
import torch.nn as nn
import torch.nn.functional as F

class NaryTreeLSTMCell(nn.Module):
    """
    N-ary Tree-LSTM cell that handles up to max_children.
    It learns separate weights for each child position, preserving order.
    Reference: Tai et al., 2015 (Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks)
    """
    def __init__(self, x_size, h_size, max_children=3):
        super(NaryTreeLSTMCell, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.max_children = max_children
        
        # Dimensions:
        # Wx maps input x to (3 + max_children) * h_size  [i, o, u, f_1...f_N]
        self.Wx = nn.Linear(x_size, (3 + max_children) * h_size)
        
        # U maps hidden states to (3 + max_children) * h_size
        # We have max_children U matrices, one for each child position.
        # U_k applies to h_k.
        self.Uh = nn.ModuleList([
            nn.Linear(h_size, (3 + max_children) * h_size)
            for _ in range(max_children)
        ])

    def forward(self, x, children_h, children_c):
        """
        x: [batch_size, x_size]
        children_h: list of [batch_size, h_size] (len <= max_children)
        children_c: list of [batch_size, h_size] (len <= max_children)
        """
        # 1. Project input x
        gates = self.Wx(x)
        
        # 2. Accumulate projections from children hidden states
        for k, h_k in enumerate(children_h):
            if k < self.max_children and h_k is not None:
                # Add U_k * h_k
                gates = gates + self.Uh[k](h_k)
                
        # 3. Split gates
        i_gate, o_gate, u_gate, *f_gates_concat = torch.split(gates, self.h_size, dim=1)
        
        i = torch.sigmoid(i_gate)
        o = torch.sigmoid(o_gate)
        u = torch.tanh(u_gate)
        
        # 4. Compute cell state c
        # c = i * u + sum(f_k * c_k)
        c = i * u
        
        f_gates = f_gates_concat # This is a tuple of tensors
        
        for k, c_k in enumerate(children_c):
            if k < self.max_children and c_k is not None:
                f_k = torch.sigmoid(f_gates[k])
                c = c + f_k * c_k
                
        # 5. Compute hidden state h
        h = o * torch.tanh(c)
        
        return h, c

class TreeLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_children=3):
        super(TreeLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tree_lstm_cell = NaryTreeLSTMCell(embedding_dim, hidden_dim, max_children)
        self.hidden_dim = hidden_dim
        self.max_children = max_children

    def forward(self, root_nodes, symbol2idx, device):
        """
        Recursive forward pass.
        
        Args:
            root_nodes: List of TreeNode objects (roots of trees).
            symbol2idx: Dict mapping rule strings to indices. (Unused here if nodes already have idx)
            device: torch.device
            
        Returns:
            outputs: [batch_size, hidden_dim]
        """
        batch_h = []
        
        for root in root_nodes:
            h, c = self._traverse(root, device)
            batch_h.append(h)
            
        return torch.cat(batch_h, dim=0) # [batch_size, hidden_dim]

    def _traverse(self, node, device):
        """
        Recursive traversal for a single tree.
        Returns: (h, c) each of shape [1, hidden_dim]
        """
        if not node:
            return None, None
            
        # 1. Process children first (Bottom-Up)
        child_hs = []
        child_cs = []
        
        for child in node.children:
            h, c = self._traverse(child, device)
            child_hs.append(h)
            child_cs.append(c)
            
        # 2. Get embedding for current node
        idx = torch.tensor([node.rule_idx], dtype=torch.long, device=device)
        x = self.embedding(idx) # [1, embedding_dim]
        
        # 3. Compute cell
        h, c = self.tree_lstm_cell(x, child_hs, child_cs)
        
        return h, c
