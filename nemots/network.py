import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, num_transplant, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.num_transplant = num_transplant
        self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim)
        self.lstm_state = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.lstm_seq = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.dist_out = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + num_transplant - 2)
        self.value_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq, state_idx, need_embeddings=True):
        if need_embeddings:
            state = self.embedding_table(state_idx.long())
        else:
            state = state_idx

        seq = seq.unsqueeze(-1)
        out_state, _ = self.lstm_state(state)
        out_seq, _ = self.lstm_seq(seq)

        out = torch.cat([out_state[:, -1, :], out_seq[:, -1, :]], dim=-1)
        out = self.mlp(out)
        raw_dist_out = self.dist_out(out)
        raw_dist_out = torch.where(torch.isnan(raw_dist_out), torch.zeros_like(raw_dist_out), raw_dist_out)
        value_out = self.value_out(out)
        return raw_dist_out, value_out
    
    def compute_quantile_loss(self, predictions, targets, q_low=0.25, q_high=0.75):
        """
        计算分位数损失 (Pinball Loss)
        
        Args:
            predictions: 预测值 [batch_size, seq_len] 或 [num_samples, seq_len]
            targets: 真实值 [seq_len]
            q_low: 低分位数 (默认0.25)
            q_high: 高分位数 (默认0.75)
            
        Returns:
            torch.Tensor: 总的分位数损失
        """
        import numpy as np
        
        # 确保输入是torch.Tensor
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32, device=next(self.parameters()).device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32, device=next(self.parameters()).device)
        
        # 如果predictions是2D，计算分位数
        if predictions.dim() == 2 and predictions.shape[0] > 1:
            # 计算Q25和Q75分位数
            q25_pred = torch.quantile(predictions, q_low, dim=0)
            q75_pred = torch.quantile(predictions, q_high, dim=0)
        else:
            # 如果只有一个预测，直接使用
            q25_pred = predictions.flatten()
            q75_pred = predictions.flatten()
        
        # 确保targets维度匹配
        if targets.dim() > 1:
            targets = targets.flatten()
        
        # 调整维度匹配
        min_len = min(len(q25_pred), len(targets))
        q25_pred = q25_pred[:min_len]
        q75_pred = q75_pred[:min_len]
        targets = targets[:min_len]
        
        # 计算分位数损失 (Pinball Loss)
        def pinball_loss(y_true, y_pred, quantile):
            error = y_true - y_pred
            return torch.mean(torch.maximum(quantile * error, (quantile - 1) * error))
        
        # 计算Q25和Q75的分位数损失
        loss_q25 = pinball_loss(targets, q25_pred, q_low)
        loss_q75 = pinball_loss(targets, q75_pred, q_high)
        
        # 总损失
        total_loss = loss_q25 + loss_q75
        
        return total_loss


class PVNetCtx:
    def __init__(self, grammars, num_transplant, device):
        self.device = device
        self.grammar_vocab = ['f->A'] + grammars + ['placeholder' + str(i) for i in
                                                    range(num_transplant)]
        self.grammar_vocab_backups = copy.deepcopy(self.grammar_vocab)
        self.num_transplant = num_transplant
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.pv_net = PVNet(self.grammar_vocab, num_transplant).to(self.device)

    def policy_value(self, seq, state):
        
        assert seq.shape[0] > 1, f"seq shape error: {seq.shape}, should have at least 2 rows"
        state_list = state.split(",")
        state_idx = torch.Tensor([self.symbol2idx[item] for item in state_list]).to(self.device)
        seq = torch.Tensor(seq).to(self.device)
        raw_dist_out, value_out = self.pv_net(seq[1, :].unsqueeze(0), state_idx.unsqueeze(0))
        return raw_dist_out, value_out

    def process_state(self, state):
        unknown_counter = 0
        for i in range(len(state)):
            if state[i] not in self.grammar_vocab:
                state[i] = "placeholder" + str(unknown_counter)
                unknown_counter += 1
        return state

    def policy_value_batch(self, seqs, states):
        for idx, seq in enumerate(seqs):
            seqs[idx] = torch.Tensor(seq).to(self.device)

        states_list = []
        for idx, state in enumerate(states):
            state_list = state.split(",")
            processed_state_list = self.process_state(state_list)
            state_idx = torch.Tensor([self.symbol2idx[item] for item in processed_state_list]).to(self.device)
            state_emb = self.pv_net.embedding_table(state_idx.long())
            states_list.append(state_emb)
        max_len = max(state.shape[0] for state in states_list)
        for idx, state in enumerate(states_list):
            if state.shape[0] < max_len:
                states_list[idx] = F.pad(state, (0, 0, 0, max_len - state.shape[0]), "constant", 0)

        states = torch.stack(states_list).to(self.device)
        seqs = torch.stack(seqs).to(self.device)
        raw_dist_out, value_out = self.pv_net(seqs, states, False)
        return raw_dist_out, value_out

    def update_grammar_vocab_name(self, aug_grammars):
        for idx, grammar in enumerate(aug_grammars):
            placeholder_name = "placeholder" + str(idx)
            try:
                placeholder_index = self.grammar_vocab.index(placeholder_name)
                self.grammar_vocab[placeholder_index] = grammar
            except ValueError:
                print(f"警告: {placeholder_name} 不在语法词汇表中，跳过更新")
                continue

    def reset_grammar_vocab_name(self):
        self.grammar_vocab = copy.deepcopy(self.grammar_vocab_backups)
