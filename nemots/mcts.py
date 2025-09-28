import math
import sys
from collections import defaultdict

import numpy as np


class MCTS():
    def __init__(self, data_sample, base_grammars, aug_grammars, nt_nodes, max_len, max_module, aug_grammars_allowed,
                 func_score, exploration_rate=1 / np.sqrt(2), eta=0.999, inherited_tree=None):
        self.data_sample = data_sample
        self.base_grammars = base_grammars
        self.aug_grammars = [x for x in aug_grammars if x not in base_grammars]
        self.grammars = base_grammars + self.aug_grammars
        self.nt_nodes = nt_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed
        self.good_modules = []
        self.score = func_score
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(lambda: np.zeros(len(self.grammars)))
        self.QN = defaultdict(lambda: np.zeros(2))
        self.scale = 0
        self.eta = eta
        
        # 语法树继承机制
        self.inherited_tree = inherited_tree
        self.inherited_initialized = False
        
        # 如果有继承的语法树，初始化为good_modules
        if self.inherited_tree is not None:
            self._initialize_inherited_tree()
    
    def _initialize_inherited_tree(self):
        """
        初始化继承的语法树
        将前一窗口的最优解作为初始的good_modules
        """
        try:
            if isinstance(self.inherited_tree, str):
                # 如果是字符串表达式，创建一个模块记录
                inherited_module = (self.inherited_tree, 0.8, self.inherited_tree)  # 给一个较高的初始分数
                self.good_modules = [inherited_module]
                self.inherited_initialized = True
                print(f"🌳 MCTS继承语法树: {self.inherited_tree}")
            elif hasattr(self.inherited_tree, '__iter__'):
                # 如果是语法树结构
                inherited_module = (str(self.inherited_tree), 0.8, str(self.inherited_tree))
                self.good_modules = [inherited_module]
                self.inherited_initialized = True
                print(f"🌳 MCTS继承语法树结构: {str(self.inherited_tree)}")
        except Exception as e:
            print(f"⚠️ 语法树继承初始化失败: {e}")
            self.inherited_tree = None
            self.inherited_initialized = False

    def valid_prods(self, Node):
        return [self.grammars.index(x) for x in [x for x in self.grammars if x.startswith(Node)]]

    def tree_to_eq(self, prods):
        seq = ['f']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq1 = seq[:ix]
                    seq2 = list(prod[3:])
                    seq3 = seq[ix + 1:]
                    seq = seq1 + seq2 + seq3
                    break
        try:
            return ''.join(seq)
        except:
            return ''

    def state_to_seq(self, state):
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq

    def state_to_onehot(self, state):
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1
        return state_oh

    def get_ntn(self, prod, prod_idx):
        return [] if prod_idx >= len(self.base_grammars) else [i for i in prod[3:] if i in self.nt_nodes]

    def get_unvisited(self, state, node):
        return [a for a in self.valid_prods(node) if self.QN[state + ',' + self.grammars[a]][1] == 0]

    def print_solution(self, solu, i_episode):
        print('Episode', i_episode, solu)

    def step(self, state, action_idx, ntn):
        action = self.grammars[action_idx]
        state = state + ',' + action
        ntn = self.get_ntn(action, action_idx) + ntn[1:]

        if not ntn:
            reward, eq = self.score(self.tree_to_eq(state.split(',')), len(state.split(',')), self.data_sample,
                                    eta=self.eta)
            return state, ntn, reward, True, eq
        else:
            return state, ntn, 0, False, None

    def rollout_with_nn(self, num_play, state_initial, ntn_initial):
        best_eq = ''
        best_r = 0
        for n in range(num_play):
            done = False
            state = state_initial

    def rollout(self, num_play, state_initial, ntn_initial):
        best_eq = ''
        best_r = 0
        for n in range(num_play):
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                if not ntn:
                    break
                valid_index = self.valid_prods(ntn[0])
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                state = next_state
                ntn = ntn_next

                if state.count(',') >= self.max_len:
                    break

            if done:
                if reward > best_r:
                    self.update_modules(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward

        return best_r, best_eq

    def update_ucb_mcts(self, state, action):
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def update_QN_scale(self, new_scale):
        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def backpropogate(self, state, action_index, reward):
        action = self.grammars[action_index]

        if self.scale != 0:
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0

        self.QN[state + ',' + action][1] += 1

        while state:
            if self.scale != 0:
                self.QN[state][0] += reward / self.scale
            else:
                self.QN[state][0] += 0

            self.QN[state][1] += 1
            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)

            if ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def get_policy1(self, nA, state, node):
        valid_action = self.valid_prods(node)
        policy_valid = []
        sum_ucb = sum(self.UCBs[state][valid_action])

        for a in valid_action:
            policy_mcts = self.UCBs[state][a] / sum_ucb
            policy_valid.append(policy_mcts)

        if len(set(policy_valid)) == 1:
            A = np.zeros(nA)
            A[valid_action] = float(1 / len(valid_action))
            return A

        A = np.zeros(nA, dtype=float)
        best_action = valid_action[np.argmax(policy_valid)]
        A[best_action] += 0.8
        A[valid_action] += float(0.2 / len(valid_action))
        return A

    def get_policy3(self, nA, UC, seq, state, network, softmax=True):
        policy, value = network.policy_value(seq, state)
        policy = policy.cpu().detach().squeeze(0).numpy()
        
        if nA == 0:
            pass
        policy = self.softmax(policy[:nA]) if softmax else policy[:nA]
        return policy, value

    def modify_UCB(self, probs, state):
        N_parent = self.QN[state][1]
        ucbs = [
            score * math.sqrt(N_parent) / (self.QN[state + ',' + action][1] + 1) +
            max(self.QN[state + ',' + action][0], 0)
            for action, score in zip(self.grammars, probs)
        ]
        return self.softmax(np.asarray(ucbs))

    def update_modules(self, state, reward, eq):
        module = state[5:]
        if state.count(',') <= self.max_module:
            if not self.good_modules:
                self.good_modules = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.good_modules]:
                if len(self.good_modules) < self.max_aug:
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.good_modules[0][1]:
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key=lambda x: x[1])

    def run(self, seq, num_episodes, network, num_play=50, print_flag=False, print_freq=100, use_network=True, alpha=None, buffer_size=None, current_buffer_size=None):
        nA = len(self.grammars)
        network.update_grammar_vocab_name(self.aug_grammars)

        # 动态融合系数alpha
        if alpha is None:
            if buffer_size is not None and current_buffer_size is not None:
                alpha = min(1.0, current_buffer_size / buffer_size)
            else:
                alpha = 1.0 if use_network else 0.0

        states = []
        reward_his = []
        best_solution = ('nothing', 0)

        state_records = []
        seq_records = []
        policy_records = []
        value_records = []

        for i_episode in range(1, num_episodes + 1):
            if (i_episode) % print_freq == 0 and print_flag:
                print(f"\rEpisode {i_episode}/{num_episodes}, current best reward {best_solution[1]}, alpha={alpha:.3f}", end="")
                sys.stdout.flush()

            state = 'f->A'
            ntn = ['A']
            UC = self.get_unvisited(state, ntn[0])

            while not UC:
                # 融合policy: alpha*NN + (1-alpha)*UCB1
                policy_nn, value_nn = self.get_policy3(nA, UC, seq, state, network, softmax=True)
                policy_ucb = self.get_policy1(nA, state, ntn[0])
                policy = alpha * policy_nn + (1 - alpha) * policy_ucb
                policy = np.clip(policy, 1e-8, 1)  # 防止全零
                policy = policy / policy.sum()
                
                try:
                    action = np.random.choice(np.arange(nA), p=policy)
                except ValueError:
                    action = np.random.choice(np.arange(nA), p=np.full(nA, 1 / nA))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)

                if state not in states:
                    states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])

                    if state.count(',') >= self.max_len:
                        UC = []
                        self.backpropogate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break
                else:
                    UC = []
                    # 融合value: alpha*NN + (1-alpha)*rollout
                    if alpha > 0:
                        value_nn = float(value_nn)
                    else:
                        value_nn = 0.0
                    if alpha < 1:
                        value_rollout, _ = self.rollout(num_play, next_state, ntn_next)
                    else:
                        value_rollout = 0.0
                    reward = alpha * value_nn + (1 - alpha) * value_rollout
                    
                    if reward > best_solution[1]:
                        self.update_modules(next_state, reward, eq)
                        self.update_QN_scale(reward)
                        best_solution = (eq, reward)

                    self.backpropogate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            if UC:
                # 融合policy
                policy_nn, value_nn = self.get_policy3(nA, UC, seq, state, network, softmax=True)
                policy_ucb = self.get_policy1(nA, state, ntn[0])
                policy = alpha * policy_nn + (1 - alpha) * policy_ucb
                policy = np.clip(policy, 1e-8, 1)
                policy = policy / policy.sum()
                
                try:
                    action = np.random.choice(np.arange(nA), p=policy)
                except ValueError:
                    action = np.random.choice(np.arange(nA), p=np.full(nA, 1 / nA))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                if eq is not None:
                    state_records.append(state)
                    seq_records.append(seq)
                    policy_records.append(policy)
                    value_records.append(float(value_nn))

                if not done:
                    # 融合value
                    if alpha > 0:
                        value_nn = float(value_nn)
                    else:
                        value_nn = 0.0
                    if alpha < 1:
                        value_rollout, _ = self.rollout(num_play, next_state, ntn_next)
                    else:
                        value_rollout = 0.0
                    reward = alpha * value_nn + (1 - alpha) * value_rollout
                    
                    if state not in states:
                        states.append(state)

                if reward > best_solution[1]:
                    self.update_QN_scale(reward)
                    best_solution = (eq, reward)

                self.backpropogate(state, action, reward)
                reward_his.append(best_solution[1])

        return reward_his, best_solution, self.good_modules, zip(state_records, seq_records, policy_records,
                                                                 value_records)

    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
