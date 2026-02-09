import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
  ###！！！ network.py 定义了两个核心类：
  #1.  PVNet: 一个继承自 torch.nn.Module 的实际神经网络模型。它包含嵌入层、LSTM层和全连接层，负责具体的计算。
  #2.  PVNetCtx: 一个上下文/控制器类。它包装了 PVNet，并为项目中的其他部分（主要是
  #mcts.py）提供了一个干净、统一的接口。它负责处理数据格式的转换（例如，将字符串表达式转换为网络能理解的张量）和词汇表的管理。

  #这里主要是借助torch.nn库进行快速的构建和训练神经网络
  #module是该库最核心的部分，提供深度学习中常用的各类层
class PVNet(nn.Module):
    def __init__(self, grammar_vocab, num_transplant, hidden_dim=16):
        #nn.init模块可提供各种参数的初始化方法
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.num_transplant = num_transplant
        self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim) #嵌入层
        #创建LSTM长短期记忆网络
        self.lstm_state = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.lstm_seq = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        #创建一个多层感知机（MLP)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True))

        self.dist_out = nn.Linear(hidden_dim * 2, len(self.grammar_vocab) + num_transplant - 2)
        self.value_out = nn.Linear(hidden_dim * 2, 1)
        self.profit_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq, state_idx, need_embeddings=True):
        #定义了数据在网络中的流动方式（前向传播）
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
        
        # 🎯 架构级变体注入：仅在消融实验模式下生效
        value_out = self.value_out(out)
        profit_out = self.profit_out(out)
        
        # 检查是否为消融实验模式
        import os
        ablation_mode = os.getenv('ABLATION_EXPERIMENT_MODE', '').lower() == 'true'
        return raw_dist_out, value_out, profit_out

#与mcts统一的接口 真正调用的part
class PVNetCtx:
   # 定义一个名为PVNetCtx的类。它不是一个nn.Module，而是一个控制器或包装器，负责管理PVNet实例并提供方便的接口。
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
        state_list = state.split(",")
        state_idx = torch.Tensor([self.symbol2idx[item] for item in state_list]).to(self.device)
        seq = torch.Tensor(seq).to(self.device)
        raw_dist_out, value_out, profit_out = self.pv_net(seq[1, :].unsqueeze(0), state_idx.unsqueeze(0))
        return raw_dist_out, value_out, profit_out
   #返回概率和价值

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
        raw_dist_out, value_out, profit_out = self.pv_net(seqs, states, False)
        return raw_dist_out, value_out, profit_out

    def update_grammar_vocab_name(self, aug_grammars):
        # Rebuild the vocabulary with the base and augmented grammars
        self.grammar_vocab = ['f->A'] + self.base_grammars + aug_grammars
        
        # Rebuild the symbol-to-index mapping
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        
        # Re-initialize the network to resize the layers according to the new vocabulary size
        self.pv_net = PVNet(self.grammar_vocab, self.num_transplant).to(self.device)
        
        # 🎯 重建网络后恢复变体模式设置
        if hasattr(self, '_variant_mode'):
            self.pv_net._variant_mode = self._variant_mode
            print(f"DEBUG: Network rebuilt with variant mode: {self._variant_mode}")
        else:
            print(f"DEBUG: Network rebuilt without variant mode")
            # 检查是否应该从环境变量推断变体模式
            import os
            if os.getenv('ABLATION_EXPERIMENT_MODE', '').lower() == 'true':
                print(f"DEBUG: Ablation mode detected but no variant mode set on PVNetCtx")
        
        print(f"DEBUG: Network rebuilt. New vocab size: {len(self.grammar_vocab)}, New policy output size: {self.pv_net.dist_out.out_features}")

    def reset_grammar_vocab_name(self):
        self.grammar_vocab = copy.deepcopy(self.grammar_vocab_backups)
