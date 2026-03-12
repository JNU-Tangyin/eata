import math
import sys
from collections import defaultdict

import numpy as np


class MCTS():
    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def __init__(self, data_sample, base_grammars, aug_grammars, nt_nodes, max_len, max_module, aug_grammars_allowed,
                 func_score, exploration_rate=1 / np.sqrt(2), eta=0.999, initial_tree=None):
        # 类的构造函数，用于初始化MCTS搜索所需的所有参数和数据结构。 这里类的构造就是提供一个模板
        #这里构造类函数，在model中调用，这些参数在model中有对应的值，这里相当于一个模版
        self.data_sample = data_sample  # 当前时间窗口的(X, y)数据，用于最终评估表达式的分数。
        self.base_grammars = base_grammars # 基础语法规则列表，例如 ['A->A+A', 'A->x0']。在model中有具体的定义
        self.aug_grammars = [x for x in aug_grammars if x not in base_grammars]
      #  从传入的增强语法中，过滤掉已经存在于基础语法中的规则，避免重复
        self.grammars = base_grammars + self.aug_grammars #合并基础语法和增强语法，形成当前MCTS搜索可用的完整语法库。
        self.nt_nodes = nt_nodes # 非终结符列表，通常是 ['A']，代表表达式中可以被进一步展开的部分
        self.max_len = max_len  # 生成表达式的最大长度（即最多使用多少条语法规则）
        self.max_module = max_module # 一个子表达式（module）可以被接受的最大长度。
        self.max_aug = aug_grammars_allowed # 最多允许存储多少个优秀的子表达式（modules）。
        self.good_modules = [] # 用于存储搜索过程中发现的优秀子表达式（(module_str, reward, eq_str)）
        self.score = func_score # 从外部传入的评分函数，用于计算一个完整表达式的最终得分。 model中用的1是score文件
        self.exploration_rate = exploration_rate # UCT公式中的探索常数C。
        self.UCBs = defaultdict(lambda: np.zeros(len(self.grammars)))#
       # 存储UCT值的字典。键是状态(state)，值是一个数组，数组的每个元素对应一条语法规则的UCT值。
        self.QN = defaultdict(lambda: np.zeros(2)) #记录每次mcts中每个节点的Q值和N值
        self.scale = 0  # 用于归一化奖励的缩放因子
        self.eta = eta  # 评分函数中使用的一个参数。
        self.initial_tree = initial_tree

#这里也不用动
    def valid_prods(self, Node):
        # 功能：给定一个非终结符(Node, e.g., 'A')，返回所有可以用于展开它的语法规则的索引。
        return [self.grammars.index(x) for x in [x for x in self.grammars if x.startswith(Node)]]
    # 这是一个两层列表推导：
    # 1. [x for x in self.grammars if x.startswith(Node)]：筛选出所有以Node开头的语法规则。
    # 2. [self.grammars.index(x) for x in ...]：获取这些规则在self.grammars完整列表中的索引。

  #这里也不用变 我门要变的是他的action ，时进行修补而不是重新的生成 这里可以也要稍作修改
    #要稍微修改 这里的序列是上次继承下来的话，就不用初始化序列了，是不是用一次就OK了 后面的都已经转化完成了
    def tree_to_eq(self, prods):
        # 功能：将一个语法规则序列(prods)转换成一个可计算的数学表达式字符串。
        seq = ['f'] # 初始化序列，'f'是语法的起始符号。
        for prod in prods: # 遍历传入的每条语法规则。
            if str(prod[0]) == 'Nothing': # 如果遇到'Nothing'规则，表示表达式构建提前终止
                break
            for ix, s in enumerate(seq): # 遍历当前表达式序列中的每个符号。
                if s == prod[0]: # 找到第一个与当前规则左侧匹配的非终结符。
                    seq1 = seq[:ix]  # 截取匹配位置之前的部分。
                    seq2 = list(prod[3:]) # 获取规则右侧的符号序列 (e.g., for 'A->A+A', this is ['A','+', 'A'])。
                    seq3 = seq[ix + 1:]  # 截取匹配位置之后的部分
                    seq = seq1 + seq2 + seq3 # 将三部分拼接起来，完成一次替换。
                    break
        try:
            return ''.join(seq) # 将最终的符号列表拼接成字符串
        except:
            return ''  # 如果过程中出现异常（例如序列中含有非字符串元素），返回空字符串。
#

    def state_to_seq(self, state):
        #将状态转化为序列
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq

    def state_to_onehot(self, state):
        # 编码为神经网络可以识别的one-hot矩阵
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1
        return state_oh

    def get_ntn(self, prod, prod_idx):# 功能：获取一条产生式规则(prod)右侧的所有非终结符(Non-Terminal Nodes)。

        return [] if prod_idx >= len(self.base_grammars) else [i for i in prod[3:] if i in self.nt_nodes]
        # 如果规则索引超出了基础语法的范围，说明它是一个增强语法(module)，我们视其为终结符，不产生新的非终 结符，返回[]。# 否则，遍历规则右侧的每个符号，筛选出属于非终结符的那些

    def get_unvisited(self, state, node):
        # 功能：获取当前状态(state)下所有未被访问过的子节点（动作）。
        return [a for a in self.valid_prods(node) if self.QN[state + ',' + self.grammars[a]][1] == 0]

        # 1. self.valid_prods(node): 获取所有合法的下一步动作（规则索引）。
        # 2. self.QN[state + ',' + self.grammars[a]][1] == 0:检查执行该动作后的新状态的访问次数(N) 是否为0。
    # 3. 列表推导筛选出所有访问次数为0的动作。

    def print_solution(self, solu, i_episode):
        print('Episode', i_episode, solu)

    def step(self, state, action_idx, ntn):
        # 功能：在MCTS中执行一步，即应用一个语法规则
        action = self.grammars[action_idx] # 根据动作索引获取语法规则字符串。
        state = state + ',' + action # 将规则拼接到当前状态字符串后面，形成新状态。
        ntn = self.get_ntn(action, action_idx) + ntn[1:]
       # 更新待展开的非终结符列表：将新规则产生的非终结符加到列表头部，并移除刚刚被展开的那个

        if not ntn: # 如果待展开列表为空，说明表达式已经构建完成。
            reward, eq = self.score(self.tree_to_eq(state.split(',')), len(state.split(',')), self.data_sample,
                                    eta=self.eta)
            # 调用评分函数计算最终得分(reward)和表达式(eq)。
            return state, ntn, reward, True, eq  # 返回新状态、空列表、奖励、完成标志(True)和表达式。
        else: # 如果表达式未构建完成。
            return state, ntn, 0, False, None  #返回新状态、更新后的ntn列表、0奖励、未完成标志(False)和None。

    def rollout(self, num_play, state_initial, ntn_initial):
        # 功能：执行N次(num_play)蒙特卡洛随机模拟（Rollout）。
        best_eq = ''  # 记录这N次模拟中找到的最佳表达式。
        best_r = 0# 记录最佳表达式对应的奖励。先默认空值
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
        # 功能：计算并更新UCT值。这是UCT算法的核心公式。
        def UCT(self, state, next_state):
            Q_child = self.QN[next_state][0]
            N_parent = self.QN[state][1]
            N_child = self.QN[next_state][1]
            # 使用原始的UCT公式，不进行人为放大
            exploration_term = self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)
            uct_value = Q_child / N_child + exploration_term
            
            # 调试信息：每100次调用打印一次
            if not hasattr(self, '_uct_call_count'):
                self._uct_call_count = 0
            self._uct_call_count += 1
            if self._uct_call_count % 100 == 0:
                print(f"🔍 UCT调试 (第{self._uct_call_count}次): exploration_rate={self.exploration_rate:.6f}, exploration_term={exploration_term:.6f}, uct_value={uct_value:.6f}")
            
            return uct_value

        next_state = state + ',' + action
        return UCT(self, state, next_state)

    def update_QN_scale(self, new_scale):
        # 功能：当找到一个新的全局最优解时，重新缩放所有节点的Q值
        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def backpropogate(self, state, action_index, reward):
        # 功能：执行反向传播，将一次模拟的奖励(reward)更新回从叶节点到根节点的整条路径。
        #以上这些都是事先定义的函数，主运行程序还没到
        action = self.grammars[action_index] # 获取动作（规则）字符串

        if self.scale != 0:  # 如果设置了缩放因子
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0

        self.QN[state + ',' + action][1] += 1  # 将当前叶节点的访问次数N加1

        while state: # 循环，直到回溯到根节点（state为空字符串）。
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
        # 功能：根据MCTS的UCT值计算一个策略。
        valid_action = self.valid_prods(node)  # 获取所有合法动作
        policy_valid = []  # 存储合法动作的策略值。
        sum_ucb = sum(self.UCBs[state][valid_action])  # 计算所有合法动作的UCT值之和。

        if sum_ucb == 0: # if all UCBs are 0, return a uniform policy
            A = np.zeros(nA)
            if len(valid_action) > 0:
                A[valid_action] = float(1 / len(valid_action))
            return A
        for a in valid_action: # 遍历合法动作。
            policy_mcts = self.UCBs[state][a] / sum_ucb  # 计算每个动作的策略概率（UCT值归一化）。
            policy_valid.append(policy_mcts)

        if len(set(policy_valid)) == 1:  # 如果所有合法动作的UCT值都一样（例如，都是0）。
            A = np.zeros(nA) # 创建一个全零策略向量
            A[valid_action] = float(1 / len(valid_action))  # 将合法动作的概率设为均等。
            return A # 返回均匀策略。

        A = np.zeros(nA, dtype=float)
        best_action = valid_action[np.argmax(policy_valid)]  # 找到UCT值最高的那个动作。
        
        # 🔧 修复：当exploration_rate=0时，使用纯贪心策略（确定性选择最优动作）
        # 这确保NoExplore变体真正禁用探索，与Full变体产生不同的决策路径
        if hasattr(self, 'exploration_rate') and self.exploration_rate == 0.0:
            # NoExplore变体：100%选择最优动作（纯利用，无探索）
            A[best_action] = 1.0
            
            # 调试信息：每100次调用打印一次
            if not hasattr(self, '_greedy_policy_count'):
                self._greedy_policy_count = 0
            self._greedy_policy_count += 1
            if self._greedy_policy_count % 100 == 0:
                print(f"🎯 [NoExplore] 纯贪心策略 (第{self._greedy_policy_count}次): 100%选择最优动作 {best_action}")
        else:
            # 其他变体：探索性策略（80%最优，20%探索）
            A[best_action] += 0.8  # 将80%的概率分配给最好的动作。
            A[valid_action] += float(0.2 / len(valid_action)) #将剩下20%的概率均匀分配给所有合法动作（包括最好的那个）。
        
        return A  # 返回策略
#与network的借口哦
    def get_policy3(self, nA, UC, seq, state, network, softmax=True):
        # 功能：调用神经网络获取策略(policy)、价值(value)和盈利预测(profit)。
        policy, value, profit = network.policy_value(seq, state) # 调用网络的前向传播函数。
        policy = policy.cpu().detach().squeeze(0).numpy()

        # 保存profit预测用于后续分析或调试
        self.last_profit_pred = profit.cpu().detach().squeeze(0).numpy()

        if nA == 0:   # 如果没有可用动作
            return np.array([]), value, profit

        policy = self.softmax(policy[:nA]) if softmax else policy[:nA]
        return policy, value, profit

    def modify_UCB(self, probs, state):
        # 这是一个备用/未使用的UCT计算函数，它融合了神经网络的先验概率(probs)，更接近AlphaGo的PUCT公式
        N_parent = self.QN[state][1]
        ucbs = [
            score * math.sqrt(N_parent) / (self.QN[state + ',' + action][1] + 1) +
            max(self.QN[state + ',' + action][0], 0)
            for action, score in zip(self.grammars, probs)
        ]
        return self.softmax(np.asarray(ucbs))

    def update_modules(self, state, reward, eq):
        # 🔧 检测 skip_memory 参数：NoMem 变体跳过模块库更新
        if hasattr(self, '_variant_skip_memory') and self._variant_skip_memory:
            return  # 跳过模块库更新
        
        # 功能：维护一个高质量子表达式（module）的列表。 会进行更新
        module = state[5:]  # 从状态字符串中截取模块部分（去掉'f->A,'）
        if state.count(',') <= self.max_module: # 如果模块长度在允许范围内。
            if not self.good_modules:  # 如果列表是空的。
                self.good_modules = [(module, reward, eq)] # 直接添加。
            elif eq not in [x[2] for x in self.good_modules]: # 如果这个表达式还没被记录过。
                if len(self.good_modules) < self.max_aug:  # 如果列表还没满
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key=lambda x: x[1]) # 添加并按奖励排序
                else:  # 如果列表满了。
                    if reward > self.good_modules[0][1]: # 如果新模块的奖励高于列表中最差的那个。
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key=lambda x: x[1])
    # 替换掉最差的，并重新排序

#——————这里才是主程序———— model中调用mcts中的run时，对应的参数都会有给出的。
    def run(self, seq, num_episodes, network, num_play=50, print_flag=False, print_freq=100, use_network=True, alpha=None, buffer_size=None, current_buffer_size=None):
        nA = len(self.grammars)  # 获取当前总的动作数量（语法规则数量）。
        # 功能：MCTS的主运行函数，执行完整的“选择-展开-模拟-反向传播”循环。
        network.update_grammar_vocab_name(self.aug_grammars)  # 通知神经网络，词汇表（语法）已经更新。

        # 🔍 调试：打印接收到的参数
        print(f"🔍 [MCTS.run] 接收参数: use_network={use_network}, alpha={alpha}, exploration_rate={self.exploration_rate}")
        
        # 动态计算融合系数alpha
        if alpha is None:
            if buffer_size is not None and current_buffer_size is not None:
                # 使用与model.py一致的ceiling逻辑，避免用maxlen(10240)作为分母导致alpha≈0
                # 假设train_size约为buffer_size/320，ceiling=train_size*16
                estimated_train_size = max(32, buffer_size // 320)
                alpha_ceiling = estimated_train_size * 16
                alpha = min(0.5, current_buffer_size / max(1, alpha_ceiling))
            else:
                alpha = 1.0 if use_network else 0.0
        else:
            # 使用变体传入的alpha参数，不进行覆盖
            print(f"   🔧 使用变体指定的 alpha={alpha}")
            pass
        
        # 🔍 调试：打印最终使用的alpha
        print(f"🔍 [MCTS.run] 最终alpha={alpha}, use_network={use_network}")

        states = [] # 记录访问过的状态。
        reward_his = []  # 记录每个episode后的最佳奖励历史。
        best_solution = ('nothing', 0)  # 记录全局最优解（表达式，奖励）。
        best_solution_node = None  # 用于存储最优树节点

        state_records = []
        seq_records = []
        policy_records = []
        value_records = []
        reward_records = []  # 新增：reward记录
#——————这里就要动手了 我们不要每次都生成，而是进行修补
        for i_episode in range(1, num_episodes + 1):
            # 主循环，执行num_episodes次完整的树搜索。
            if (i_episode) % print_freq == 0 and print_flag:
                print(f"\rEpisode {i_episode}/{num_episodes}, current best reward {best_solution[1]}, alpha={alpha:.3f}", end="")
                sys.stdout.flush()

            # --- 每个Episode开始，根据initial_tree设置起点 ---
            if self.initial_tree is not None and i_episode == 1:
                state = self.initial_tree.get('state', 'f->A')
                ntn = self.initial_tree.get('ntn', ['A'])
                # 重置继承节点的统计信息，以适应新数据
                self.QN[state] = np.zeros(2)

                # [FIX] 增加对继承已完成树的处理逻辑 (评估和挑战)
                if not ntn:
                    # 1. 继承的是完整公式，立即用新数据评估它
                    reward, eq = self.score(self.tree_to_eq(state.split(',')), len(state.split(',')), self.data_sample, eta=self.eta)
                    
                    # 2. 将其设为本次运行的初始“最佳解”
                    if reward > best_solution[1]:
                        best_solution = (eq, reward)
                        best_solution_node = {'state': state, 'ntn': ntn}
                    
                    # 3. 重置搜索，从零开始，挑战这个“最佳解”
                    state = 'f->A'
                    ntn = ['A']
            else:
                # 原始逻辑，从零开始
                state = 'f->A'
                ntn = ['A']

            UC = self.get_unvisited(state, ntn[0]) # 检查根节点是否有未访问的子节点。


             # --- 1. 选择 (Selection) --- 这里面具体的过程是可以直接用的
            while not UC:  # 当没有未访问子节点时，持续向下选择 这里子节点都有了访问，这里就可以计算UCT了
                # 🔧 修复：根据use_network决定策略计算方式
                if use_network:
                    # 调用神经网络获取策略
                    policy_nn, value_nn, profit_nn = self.get_policy3(nA, UC, seq, state, network, softmax=True)
                    
                    # 记录经验（用于训练神经网络）
                    w = 0.5
                    value_accuracy = float(value_nn.detach() if hasattr(value_nn, 'detach') else value_nn)
                    value_profit = float(profit_nn.detach() if hasattr(profit_nn, 'detach') else profit_nn)
                    value_fused_nn = w * value_accuracy + (1 - w) * value_profit
                    state_records.append(state)
                    seq_records.append(seq)
                    policy_records.append(policy_nn)
                    value_records.append(value_fused_nn)
                    
                    # 获取UCB策略并融合
                    policy_ucb = self.get_policy1(nA, state, ntn[0])
                    policy = alpha * policy_nn + (1 - alpha) * policy_ucb
                else:
                    # NoNN变体：完全不调用神经网络，只使用UCB
                    # 🔧 修复：初始化value_nn和profit_nn为None，避免后续引用未定义变量
                    value_nn = None
                    profit_nn = None
                    policy_ucb = self.get_policy1(nA, state, ntn[0])
                    policy = policy_ucb
                
                # 调试信息
                import os
                ablation_mode = os.getenv('ABLATION_EXPERIMENT_MODE', '').lower() == 'true'
                if ablation_mode:
                    if hasattr(self, 'exploration_rate') and self.exploration_rate == 0.0:
                        print(f"🔍 [消融实验] NoExplore变体：exploration_rate=0, UCT只有利用项")
                    print(f"🔍 [消融实验] 策略融合：NN权重={alpha:.3f}, UCB权重={1-alpha:.3f}")
                
                policy = np.clip(policy, 1e-8, 1)  # 防止全零
                policy = policy / policy.sum()  # 重新归一化
                
                # 🔧 修复：NoExplore变体在融合后也应该使用纯贪心策略
                # 将融合后的概率分布转换为确定性策略（100%选择最优动作）
                
                # 调试：检查条件
                has_attr = hasattr(self, 'exploration_rate')
                if has_attr:
                    exp_rate = self.exploration_rate
                    is_zero = (exp_rate == 0.0)
                    
                    if not hasattr(self, '_debug_logged'):
                        print(f"🔍 [调试] exploration_rate属性存在: {has_attr}, 值: {exp_rate}, 是否为0: {is_zero}")
                        self._debug_logged = True
                    
                    if is_zero:
                        best_action_idx = np.argmax(policy)
                        old_policy = policy.copy()
                        policy = np.zeros(nA)
                        policy[best_action_idx] = 1.0
                        
                        # 调试日志：只打印一次
                        if not hasattr(self, '_noexplore_greedy_logged'):
                            print(f"🎯 [NoExplore] 融合后转换为纯贪心：100%选择最优动作 (exploration_rate={exp_rate})")
                            print(f"   原策略: {old_policy[:min(5, len(old_policy))]}")
                            print(f"   新策略: {policy[:min(5, len(policy))]}")
                            self._noexplore_greedy_logged = True
                
                try:
                    action = np.random.choice(np.arange(nA), p=policy)  #根据融合后的策略选择一个动作。
                except ValueError:  # 如果policy因为浮点数问题加和不为1。
                    action = np.random.choice(np.arange(nA), p=np.full(nA, 1 / nA))  # 使用均匀策略。
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn) # 执行一步。

                if state not in states:
                    states.append(state)

                if not done: # 如果还未生成完整表达式。
                    state = next_state  # 进入下一状态。
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])  # 检查新状态是否有未访问的子节点。

                    if state.count(',') >= self.max_len:
                        UC = [] #如果超出maxlen 强制认为没有未访问节点 进行终止
                        self.backpropogate(state, action, 0) # 以0奖励进行反向传播
                        reward_his.append(best_solution[1])  # 记录当前最佳奖励。
                        break
                else: # 如果生成了完整表达式 (done=True)。
                    UC = []  # 强制认为没有未访问节点
                    # 🔧 修复：done=True时，reward已经是self.step()返回的真实得分，不应该重新计算
                    # 保持reward不变，使用self.score()计算的真实值
                    
                    if reward > best_solution[1]:  # 如果发现了新的全局最优解。
                        self.update_modules(next_state, reward, eq)  # 更新模块库。
                        self.update_QN_scale(reward)# 更新Q值缩放因子。
                        best_solution = (eq, reward) # 更新最优解
                        best_solution_node = {'state': next_state, 'ntn': ntn_next}

                   # --4.反向传播(Backpropagation) - --
                    self.backpropogate(state, action, reward) # 将融合后的奖励反向传播。
                    reward_his.append(best_solution[1])  # 记录当前最佳奖励
                    break

            if UC:#如果发现了有未访问的子节点的节点
                # 🔧 修复：根据use_network决定策略计算方式
                if use_network:
                    # 调用神经网络获取策略
                    policy_nn, value_nn, profit_nn = self.get_policy3(nA, UC, seq, state, network, softmax=True)
                    
                    # 记录经验（用于训练神经网络）
                    w = 0.5
                    value_accuracy = float(value_nn.detach() if hasattr(value_nn, 'detach') else value_nn)
                    value_profit = float(profit_nn.detach() if hasattr(profit_nn, 'detach') else profit_nn)
                    value_fused_nn = w * value_accuracy + (1 - w) * value_profit
                    state_records.append(state)
                    seq_records.append(seq)
                    policy_records.append(policy_nn)
                    value_records.append(value_fused_nn)
                    
                    # 获取UCB策略并融合
                    policy_ucb = self.get_policy1(nA, state, ntn[0])
                    policy = alpha * policy_nn + (1 - alpha) * policy_ucb
                else:
                    # NoNN变体：完全不调用神经网络，只使用UCB
                    # 🔧 修复：初始化value_nn和profit_nn为None，避免后续引用未定义变量
                    value_nn = None
                    profit_nn = None
                    policy_ucb = self.get_policy1(nA, state, ntn[0])
                    policy = policy_ucb
                
                # 调试信息
                import os
                ablation_mode = os.getenv('ABLATION_EXPERIMENT_MODE', '').lower() == 'true'
                if ablation_mode:
                    if hasattr(self, 'exploration_rate') and self.exploration_rate == 0.0:
                        print(f"🔍 [消融实验] NoExplore变体(展开)：exploration_rate=0, UCT只有利用项")
                    print(f"🔍 [消融实验] 策略融合(展开)：NN权重={alpha:.3f}, UCB权重={1-alpha:.3f}")
                
                policy = np.clip(policy, 1e-8, 1)  # 防止全零
                policy = policy / policy.sum()  # 重新归一化
                
                # 🔧 修复：NoExplore变体在展开阶段也应该使用纯贪心策略
                # 将融合后的概率分布转换为确定性策略（100%选择最优动作）
                if hasattr(self, 'exploration_rate') and self.exploration_rate == 0.0:
                    best_action_idx = np.argmax(policy)
                    policy = np.zeros(nA)
                    policy[best_action_idx] = 1.0
                    
                    if ablation_mode and not hasattr(self, '_noexplore_expansion_logged'):
                        print(f"🎯 [NoExplore] 展开阶段转换为纯贪心：100%选择最优动作")
                        self._noexplore_expansion_logged = True
                
                try:
                    action = np.random.choice(np.arange(nA), p=policy)  #根据融合策略选择一个动作进行展开。
                except ValueError:
                    action = np.random.choice(np.arange(nA), p=np.full(nA, 1 / nA))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn) # 执行展开。 对未访问的节点 进行展开
                # 🔧 方案4改进版：已在调用神经网络时记录经验，此处不再重复记录

                if not done:  # 如果展开后还未结束。 要下一步继续选择
                    # value estimation统一使用rollout，与use_network/alpha无关
                    # alpha只影响policy融合（搜索方向），不影响value estimation
                    # Full vs NoNN的区别：policy引导方式（有NN vs 纯UCB），value相同
                    reward, _ = self.rollout(num_play, next_state, ntn_next)

                    if state not in states:
                        states.append(state)

                if reward > best_solution[1]:  # 如果发现了新的全局最优解
                    self.update_QN_scale(reward)
                    best_solution = (eq, reward)
                    best_solution_node = {'state': next_state, 'ntn': ntn_next}

                # --- 4. 反向传播 (Backpropagation) ---
                self.backpropogate(state, action, reward) # 将奖励反向传播
                reward_his.append(best_solution[1])

            return best_solution_node, best_solution, self.good_modules, zip(state_records, seq_records, policy_records,
                                                                             value_records)

        # 返回：奖励历史，最优解，优秀模块库，以及用于训练的经验数据。
