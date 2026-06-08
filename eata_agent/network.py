import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_lstm import TreeLSTMEncoder
from .tree_utils import rules_to_tree

# Hardcoded for now, or pass via init
NON_TERMINALS = ['A', 'B', 'C', 'D', 'W', 'L'] 

class PVNet(nn.Module):
    def __init__(self, grammar_vocab, num_transplant, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.num_transplant = num_transplant
        
        # Replace Embedding + LSTM with TreeLSTMEncoder
        # self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim)
        # self.lstm_state = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        self.tree_encoder = TreeLSTMEncoder(
            vocab_size=len(self.grammar_vocab), 
            embedding_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            max_children=3 # Matches max arity in grammar
        )
        
        self.lstm_seq = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.dist_out = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + num_transplant - 2)
        self.value_out = nn.Linear(hidden_dim * 2, 1)
        self.profit_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq, root_nodes, need_embeddings=True):
        """
        seq: [batch_size, seq_len] (input data)
        root_nodes: List of TreeNode (length = batch_size)
        """
        # Encode Tree Structure
        # out_state: [batch_size, hidden_dim]
        # We need to pass a dummy symbol2idx or remove it from TreeLSTM signature if unused
        out_state = self.tree_encoder(root_nodes, None, seq.device)
        
        # Encode Data Sequence
        seq = seq.unsqueeze(-1) # [batch_size, seq_len, 1]
        out_seq, _ = self.lstm_seq(seq) # [batch_size, seq_len, hidden_dim]
        
        # Concatenate: Tree embedding + Last time step of data embedding
        out = torch.cat([out_state, out_seq[:, -1, :]], dim=-1)
        
        out = self.mlp(out)
        raw_dist_out = self.dist_out(out)
        raw_dist_out = torch.where(torch.isnan(raw_dist_out), torch.zeros_like(raw_dist_out), raw_dist_out)
        value_out = self.value_out(out)
        profit_out = self.profit_out(out)
        return raw_dist_out, value_out, profit_out

class PVNetCtx:
    def __init__(self, grammars, num_transplant, device):
        self.device = device
        self.base_grammars = grammars
        self.num_transplant = num_transplant
        
        # Initial setup
        self.grammar_vocab = ['f->A'] + self.base_grammars + ['placeholder' + str(i) for i in range(self.num_transplant)]
        self.grammar_vocab_backups = copy.deepcopy(self.grammar_vocab)
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.pv_net = PVNet(self.grammar_vocab, self.num_transplant).to(self.device)

    def policy_value(self, seq, state):
        assert seq.shape[0] > 1, f"seq shape error: {seq.shape}, should have at least 2 rows"
        
        # Process State (String -> List of Rules -> Tree)
        state_list = state.split(",")
        processed_state_list = self.process_state(state_list)
        
        # Construct Tree
        root = rules_to_tree(processed_state_list, self.symbol2idx, NON_TERMINALS)
        
        # Prepare Seq
        seq = torch.Tensor(seq).to(self.device)
        # Use only the feature part (row 1 to end)? 
        # mcts.py passes input_data which is [time_idx, X_flat]. So seq[1,:] is X_flat.
        input_seq = seq[1, :].unsqueeze(0) # [1, seq_len]
        
        # Forward
        # Pass list of roots (batch size 1)
        raw_dist_out, value_out, profit_out = self.pv_net(input_seq, [root])
        return raw_dist_out, value_out, profit_out

    def process_state(self, state_list):
        # Operates on list of strings directly
        new_state = []
        unknown_counter = 0
        for item in state_list:
            if item not in self.grammar_vocab:
                new_state.append("placeholder" + str(unknown_counter))
                unknown_counter += 1
            else:
                new_state.append(item)
        return new_state

    def policy_value_batch(self, seqs, states):
        # seqs: List of arrays
        # states: List of strings (state strings)
        
        seq_tensors = []
        for seq in seqs:
            seq_tensors.append(torch.Tensor(seq).to(self.device))
        seq_batch = torch.stack(seq_tensors) # [batch_size, 2, seq_len]
        input_seq_batch = seq_batch[:, 1, :] # [batch_size, seq_len]

        roots = []
        for state in states:
            state_list = state.split(",")
            processed_state_list = self.process_state(state_list)
            root = rules_to_tree(processed_state_list, self.symbol2idx, NON_TERMINALS)
            roots.append(root)

        raw_dist_out, value_out, profit_out = self.pv_net(input_seq_batch, roots, False)
        return raw_dist_out, value_out, profit_out

    def update_grammar_vocab_name(self, aug_grammars):
        # Rebuild the vocabulary with the base and augmented grammars
        self.grammar_vocab = ['f->A'] + self.base_grammars + aug_grammars
        
        # Rebuild the symbol-to-index mapping
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        
        # Re-initialize the network to resize the layers according to the new vocabulary size
        # Note: This resets weights! In a real persistent system, we might want to extend embedding.
        self.pv_net = PVNet(self.grammar_vocab, self.num_transplant).to(self.device)
        print(f"DEBUG: Network rebuilt. New vocab size: {len(self.grammar_vocab)}, New policy output size: {self.pv_net.dist_out.out_features}")

    def reset_grammar_vocab_name(self):
        self.grammar_vocab = copy.deepcopy(self.grammar_vocab_backups)

