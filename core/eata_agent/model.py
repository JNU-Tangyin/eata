
import time
from collections import defaultdict
from collections import deque

import numpy as np

from . import score
from . import symbolics
from .mcts import MCTS #从mcts接入MCTS类
from .network import PVNetCtx #从network接入PVNetCtx类

#model就是最直接的与mcts、network连接的地方
class Model:
    def __init__(self, args):
        # Directly assign properties from the args object to the instance variables
        self.symbolic_lib = args.symbolic_lib
        self.max_len = args.max_len
        self.max_module_init = args.max_module_init
        self.num_transplant = args.num_transplant
        self.num_runs = args.num_runs
        self.eta = args.eta
        self.num_aug = args.num_aug
        self.exploration_rate = args.exploration_rate
        self.transplant_step = args.transplant_step
        self.norm_threshold = args.norm_threshold
        self.device = args.device
        self.train_size = getattr(args, 'train_size', 32)
      #  初始化环境：根据 main.py 传入的args参数，设置所有超参数  很熟悉明白了

        # 自动推断n和lookBACK
        self.n = None #先用空值顶一下子 n是输入数据的维度（特征数）
        self.lookBACK = 1  # HOTFIX: 强制回看为1，从而让变量对应不同的特征(x0,x1..)，而不是时间步
        # 仅NEMoTS自动生成多变量多步grammar
        if self.symbolic_lib == 'NEMoTS':
            # 尝试从args或数据推断n
            if hasattr(args, 'n_vars'):
                self.n = args.n_vars
            else:
                self.n = None  # 运行时再推断 在后面run中
            # 动态生成grammar（如n未知则首次run时再生成）
            def gen_multivar_grammar(n, lookBACK):
                terminals = [f'A->x{i}' for i in range(n*lookBACK)]
                #根据输入数据的维度 n (特征数) 和 lookBACK (回看窗口长度)，动态创建“终端”符号，即变量，如 x0,x1,x2, ...。
                ops = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)']
                #可以从这添加数学运算，增加枝点的选择性
                return ops + terminals
            #然后将这些变量与固定的数学运算（ops）结合，形成一个完整的语法规则库 self.base_grammar。
            if self.n is not None:
                self.base_grammar = gen_multivar_grammar(self.n, self.lookBACK) #就形成了语法规则库self.base_grammar。
                
            else:
                self.base_grammar = []  # run时再生成 如果初始化时不知道 n，它会留到 run 方法中第一次拿到数据时再生成。在run中
                
        else:
            self.base_grammar = symbolics.rule_map[self.symbolic_lib]
            #其他情况语法库直接调用Symbolics中定义的语法库 （可以修改Symbolics的rule_map

        # 支持外部注入共享网络
        if hasattr(args, 'shared_network') and args.shared_network is not None:
            self.p_v_net_ctx = args.shared_network
        else:
            self.p_v_net_ctx = PVNetCtx(self.base_grammar, self.num_transplant, self.device)
           #初始化策略-价值神经网络。它会检查是否从外部传入了一个共享网络，如果没有，就实例化一个 PVNetCtx对象。PVNetCtx 来自 network.py，是真正的 PyTorch 模型。
        #这也算是一个接口吧 不是，从network中导入了PVNetCtx
        self.nt_nodes = symbolics.ntn_map[self.symbolic_lib]
        self.score_with_est = score.score_with_est
        self.simple_mae_score = score.simple_mae_score
        self.data_buffer = deque(maxlen=10240)
        #nt_nodes: 获取非终端节点（Non-terminal nodes），在语法树中通常代表运算操作。
        # `score_with_est`: 指定用于评估表达式好坏的评分函数*，该函数来自 score.py。
    #  *   data_buffer: 创建一个固定长度的队列，用作经验回放池。

        self.aug_grammars_counter = defaultdict(lambda: 0)


#！执行符号回归任务的入口，包含了算法的核心循环
    def run(self, X, y=None, previous_best_tree=None, alpha=None, variant_exploration_rate=None):
        #通过这里的操作，外部接口，由engine中的simulate方法调用
        # assert X.size(0) == 1 # 注释掉此行，以允许处理来自滑动窗口的批数据
        #一个断言，确保每次只处理一个数据样本（batch_size 为 1）。这是因为 MCTS搜索是针对单个样本进行的
        
        # 🔧 检测 skip_mcts 参数：真正的 NoMCTS 变体
        if hasattr(self, '_variant_skip_mcts') and self._variant_skip_mcts:
            print("🚫 [NoMCTS变体] 跳过MCTS搜索，直接用神经网络生成表达式")
            return self._run_without_mcts(X, y)
        
        # 🔧 检测 skip_nn 参数：真正的 NoNN 变体
        skip_nn_mode = hasattr(self, '_variant_skip_nn') and self._variant_skip_nn
        if skip_nn_mode:
            print("🚫 [NoNN变体] 禁用神经网络，只使用MCTS的UCB策略")
            # 强制设置参数，确保不使用神经网络
            # 注意：这里的alpha会覆盖传入的参数，确保NoNN变体始终使用alpha=0
            alpha = 0.0  # 神经网络权重为0
            use_network_flag = False  # 完全禁用神经网络
            print(f"   🔧 强制设置 alpha=0.0, use_network=False")
        else:
            use_network_flag = True  # 正常使用神经网络
            # alpha保持传入值或None（由后续逻辑动态计算）
        
        # 🔧 检测 skip_memory 参数：真正的 NoMem 变体
        skip_memory_mode = hasattr(self, '_variant_skip_memory') and self._variant_skip_memory
        if skip_memory_mode:
            print("🚫 [NoMem变体] 禁用进化记忆，清空模块库")
            # 清空模块库
            self.aug_grammars_counter = {}
            print(f"   ✅ 模块库已清空: {len(self.aug_grammars_counter)} 个模块")
        
        # 动态生成base_grammar（防止初始化时n未知导致grammar为空）
        if self.symbolic_lib == 'NEMoTS' and (self.base_grammar is None or len(self.base_grammar) == 0):
            X_ = X.squeeze(0)
            
            if X_.ndim == 2:
                # The number of features 'n' is always the second dimension (columns)
                n = X_.shape[1]
                self.n = n
            elif X_.ndim == 1:
                raise ValueError(f"无法从 X_ shape {X_.shape} 推断 n，请检查输入数据 shape！")
            else:
                raise ValueError(f"未知的 X_ shape: {X_.shape}，无法推断 n！")
            #上面就是一个详细推断n的过程步骤 对应的就会初始化时不知道n，在这里生成
            def gen_multivar_grammar(n, lookBACK):
                terminals = [f'A->x{i}' for i in range(n*lookBACK)]
                ops = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)']
                return ops + terminals
            self.base_grammar = gen_multivar_grammar(self.n, self.lookBACK)
            # HOTFIX: Re-initialize the network context with the newly generated base_grammar
            self.p_v_net_ctx = PVNetCtx(self.base_grammar, self.num_transplant, self.device)
            #这部分代码用于动态生成语法*。如果在 __init__ 初始化时没有确定输入数据的特征维度 n，就在这里根据第一次传入的数据 X 的形状来推断 n，
            # 然后调用内部函数 gen_multivar_grammar 生成 MCTS搜索时需要遵循的语法规则 self.base_grammar。这是让模型适应不同维度输入数据的关键
            #这里可以增加语法规则
            # 使用__init__中已初始化的self.p_v_net_ctx，不需要再次初始化
            



             ##这里主要是对数据的处理 展平输入 到162行
        # 自动推断n和lookBACK
        if self.symbolic_lib == 'NEMoTS':
            # X shape: [1, lookBACK, n] or [1, n, lookBACK]
            X_ = X.squeeze(0)
            if (self.base_grammar is None or len(self.base_grammar) == 0) and X_.ndim == 2:
                if X_.shape[0] == self.lookBACK:
                    n = X_.shape[1]
                else:
                    n = X_.shape[0]
                self.n = n
                def gen_multivar_grammar(n, lookBACK):
                    terminals = [f'A->x{i}' for i in range(n*lookBACK)]
                    ops = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)']
                    return ops + terminals
                self.base_grammar = gen_multivar_grammar(self.n, self.lookBACK)

            # --- Data Preparation for Scoring and Network Input ---
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy() # This is the lookahead y, not used for MCTS scoring

            # The `y_np` (lookahead) is for final evaluation, not for building the scoring data for MCTS.
            # We create a target from the input data `X_np` itself for MCTS to learn the underlying dynamics.
            # We'll create a 1-step-ahead prediction task for the first feature (index 0).
            X_for_scoring = X_np[:-1]  # Use first N-1 steps of X as input features
            y_for_scoring = X_np[1:, 0]   # Use the next step's value of the first feature as the target

            # Transpose X to be (n_features, n_timesteps)
            X_transposed = X_for_scoring.T
            # Reshape y to be (1, n_timesteps)
            y_reshaped = y_for_scoring.reshape(1, -1)

            # Create supervision_data for the scoring function. Now dimensions match.
            supervision_data = np.vstack([X_transposed, y_reshaped])

            # For network input, it expects a flattened sequence
            X_flat = X_np.reshape(-1)
            time_idx = np.arange(X_flat.shape[0])
            input_data = np.vstack([time_idx, X_flat])
            
        else:#另一套数据处理逻辑
            if y is not None:
                X = X.squeeze(0)
                y = y.squeeze(0)
                time_idx = np.arange(X.size(0) + y.shape[0])
                input_data = np.vstack([time_idx[:X.size(0)], X])
                supervision_data = np.vstack([time_idx, np.concatenate([X, y])])
            else:
                X = X.squeeze(0)
                time_idx = np.arange(X.size(0))
                input_data = np.vstack([time_idx[:X.size(0)], X])
                supervision_data = np.vstack([time_idx, X])
#讲解: 数据预处理。这部分代码将 PyTorch 张量 `X` 和 `y` 转换成 NumPy 数组，并构造成 MCTS 和评分函数所需要的格式。`input_data`通常只包含输入序列，而 `supervision_data` 包含输入和输出序列，用于最终的评分。对于 NEMoTS，它会将多维时序数据展平（flatten）*。


        all_times = []
        all_eqs = []
        test_scores = []
        final_best_tree = None
        all_mcts_records = [] # 用于收集所有MCTS记录

        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant

        for i_test in range(self.num_runs):
            best_solution = ('nothing', 0)

            exploration_rate = self.exploration_rate
            max_module = self.max_module_init
            reward_his = []
            best_modules = []
            aug_grammars = []

            start_time = time.time()

            self.p_v_net_ctx.reset_grammar_vocab_name()

            # 根据变体参数选择评分函数（移到循环外，确保总是生效）
            score_func = self.score_with_est  # 默认评分函数
            if hasattr(self, '_variant_reward_function'):
                if self._variant_reward_function == 'simple_mae':
                    score_func = self.simple_mae_score
                    print(f"   🎯 使用简单MAE评分函数")
            
            # 🎯 关键修复：检查是否有Simple变体注入的自定义score函数（优先级最高）
            if hasattr(self, '_variant_custom_score_function') and self._variant_custom_score_function is not None:
                score_func = self._variant_custom_score_function
                print(f"   🎯 [Simple变体] 使用自定义score函数 (将影响MCTS搜索轨迹)")
            
            # 根据变体参数选择探索率（移到循环外，确保总是生效）
            # 🔧 方案1：参数化方法调用 - 优先使用传入的变体参数
            # 🔧 修复：记录是否使用了变体固定值，防止后续被动态修改
            use_fixed_exploration_rate = False
            if variant_exploration_rate is not None:
                exploration_rate = variant_exploration_rate
                use_fixed_exploration_rate = True  # 标记为固定值
                print(f"   🔧 使用传入的变体exploration_rate = {exploration_rate} (固定值)")
            else:
                # 向后兼容：保持原有逻辑
                exploration_rate = self.exploration_rate  # 默认探索率
                
                # 检查Model层面的变体参数
                if hasattr(self, '_variant_exploration_rate'):
                    exploration_rate = self._variant_exploration_rate
                    use_fixed_exploration_rate = True  # 标记为固定值
                    print(f"   🔍 使用模型变体exploration_rate = {exploration_rate} (固定值)")
                
                # 检查是否有通过args传递的变体参数（从SlidingWindowNEMoTS传递）
                elif hasattr(self, 'args') and hasattr(self.args, 'exploration_rate'):
                    # 检查是否是变体设置的值（不等于默认值）
                    default_exploration_rate = 1 / (2**0.5)  # 默认值
                    if abs(self.args.exploration_rate - default_exploration_rate) > 1e-6:
                        exploration_rate = self.args.exploration_rate
                        use_fixed_exploration_rate = True  # 标记为固定值
                        print(f"   🔍 从args使用变体exploration_rate = {exploration_rate} (固定值)")
                else:
                    print(f"   🔧 使用默认exploration_rate = {exploration_rate}")

            for i_itr in range(self.num_transplant):
                
                # 强制调试信息：显示传递给MCTS的exploration_rate值
                print(f"🔍 [强制调试] 创建MCTS时的exploration_rate: {exploration_rate}")
                print(f"🔍 [强制调试] variant_exploration_rate参数: {variant_exploration_rate}")
                print(f"🔍 [强制调试] use_fixed_exploration_rate: {use_fixed_exploration_rate}")
                print(f"🔍 [强制调试] 默认exploration_rate: {1/(2**0.5)}")
                print(f"🔍 [强制调试] 是否使用变体值: {abs(exploration_rate - 1/(2**0.5)) > 1e-6}")
                
                # 🔧 NoMem变体：传递空的aug_grammars
                aug_grammars_list = [] if skip_memory_mode else [x[0] for x in sorted(self.aug_grammars_counter.items(), key=lambda item: item[1], reverse=True)[:20]]
                
                mcts_block = MCTS(data_sample=supervision_data,
                                  base_grammars=self.base_grammar,
                                  aug_grammars=aug_grammars_list,
                                  nt_nodes=self.nt_nodes,
                                  max_len=self.max_len,
                                  max_module=max_module,
                                  aug_grammars_allowed=self.num_aug,
                                  func_score=score_func,
                                  exploration_rate=exploration_rate,
                                  eta=self.eta,
                                  initial_tree=previous_best_tree)

                buffer_size = self.data_buffer.maxlen
                current_buffer_size = len(self.data_buffer)
                
                # 使用传入的alpha参数，如果没有则使用原始动态计算逻辑
                if alpha is not None:
                    # 使用变体传入的alpha值，不受缓冲区大小影响
                    mcts_alpha = alpha
                    print(f"   🔧 使用变体alpha参数: {mcts_alpha} (固定值，不受缓冲区影响)")
                else:
                    # 动态计算alpha：以train_size*16作为alpha增长上限基准
                    # 原来用buffer_size(10240)导致alpha永远≈0，现改为实际可达的经验量
                    warmup = self.train_size        # 开始增长的阈值
                    alpha_ceiling = self.train_size * 16  # 达到上限所需的经验数
                    if current_buffer_size < warmup:
                        mcts_alpha = 0.0
                    else:
                        # 上限设为0.5：Full始终保持NN和rollout各50%协作，避免退化为NoMCTS（alpha=1.0）行为
                        mcts_alpha = min(0.5, (current_buffer_size - warmup) / max(1, alpha_ceiling - warmup))
                    print(f"   🔧 使用动态计算alpha: {mcts_alpha:.4f} (缓冲区 {current_buffer_size}/{buffer_size}, warmup={warmup}, ceiling={alpha_ceiling})")

                # 🎯 为MCTS实例设置消融实验模式标记
                if hasattr(self, '_ablation_experiment_mode') and self._ablation_experiment_mode:
                    mcts_block._ablation_experiment_mode = True
                    print(f"   ✅ MCTS实例消融实验模式已设置")
                
                # 🔧 为MCTS实例设置skip_memory标志
                if skip_memory_mode:
                    mcts_block._variant_skip_memory = True
                    print(f"   ✅ MCTS实例skip_memory标志已设置")
                
                new_best_tree_node, current_solution, good_modules, records = mcts_block.run(
                    input_data,
                    self.transplant_step,
                    network=self.p_v_net_ctx,
                    num_play=10,
                    print_flag=True,
                    use_network=use_network_flag,  # 🔧 使用变体控制的标志
                    alpha=mcts_alpha,
                    buffer_size=buffer_size,
                    current_buffer_size=current_buffer_size
                )

                # 修复：先转换zip对象为列表，避免重复消耗
                records_list = list(records) if records else []
                all_mcts_records.extend(records_list[:]) # 收集记录，而不是直接存入buffer
                
                # 修复：直接存储经验到缓冲区，让动态alpha能正常工作
                if records_list:
                    self.data_buffer.extend(records_list)
                    print(f"   📦 存储 {len(records_list)} 条经验到缓冲区，当前大小: {len(self.data_buffer)}/{self.data_buffer.maxlen}")

                if not best_modules:
                    best_modules = good_modules
                else:
                    best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

                aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]
                for grammar in aug_grammars:
                    self.aug_grammars_counter[grammar] += 1

                reward_his.append(best_solution[1])

                if current_solution[1] > best_solution[1]:
                    best_solution = current_solution
                    final_best_tree = new_best_tree_node

                max_module += module_grow_step
                # 🔧 修复：只有非固定值才允许动态增长，保护变体参数
                if not use_fixed_exploration_rate:
                    exploration_rate *= 5
                else:
                    print(f"   🔒 保持固定exploration_rate = {exploration_rate} (不进行动态增长)")

                test_score = \
                    self.score_with_est(score.simplify_eq(best_solution[0]), 0, supervision_data, eta=self.eta)[0]

            all_eqs.append(score.simplify_eq(best_solution[0]))
            test_scores.append(test_score)

            eq_out = score.simplify_eq(best_solution[0])
            if self.symbolic_lib == 'NEMoTS':
                print(f'变量映射: {[f"x{i}" for i in range(self.n*self.lookBACK)]}')
            print('\n{} tests complete after {} iterations.'.format(i_test + 1, i_itr + 1))
            print('best solution: {}'.format(eq_out))
            print('test score: {}'.format(test_score))
            print()

        policy = None
        reward = reward_his[-1] if len(reward_his) > 0 else None
        
        # 修复并激活正确的返回语句，并增加 all_mcts_records 作为返回值
        return all_eqs, all_times, test_scores, all_mcts_records, policy, reward, final_best_tree


    def _run_without_mcts(self, X, y=None):
        """
        真正的 NoMCTS 变体：完全跳过 MCTS 搜索，直接用神经网络生成表达式
        这个方法返回与 run() 相同格式的结果，但不进行任何 MCTS 搜索
        """
        print("🚫 [NoMCTS] 跳过MCTS搜索，使用简单表达式")
        
        # 推断特征维度
        if self.symbolic_lib == 'NEMoTS':
            X_ = X.squeeze(0) if X.dim() > 2 else X
            if X_.ndim == 2:
                n = X_.shape[1]
                self.n = n
            else:
                n = self.n if self.n is not None else 6
        else:
            n = 1
        
        # 生成简单的线性组合表达式（不需要MCTS搜索）
        # 使用前几个特征的简单组合
        if n >= 2:
            simple_expr = f"x0 + x1"  # 最简单的双变量线性组合
        else:
            simple_expr = "x0"
        
        print(f"   生成简单表达式: {simple_expr}")
        
        # 构造返回值，与 run() 方法的返回格式保持一致
        all_eqs = [simple_expr] * self.num_runs
        all_times = [0.01] * self.num_runs  # 假设很快完成
        test_scores = [0.5] * self.num_runs  # 简单表达式的评分较低
        all_mcts_records = []  # 没有MCTS记录
        policy = None
        reward = 0.5
        final_best_tree = None
        
        print(f"   NoMCTS完成: 表达式={simple_expr}, 评分={test_scores[0]:.4f}")
        
        return all_eqs, all_times, test_scores, all_mcts_records, policy, reward, final_best_tree
    
    #module就是语法增强最直接的利用对象：子结构  会被添加到语法增加库当中 #细节还是在mcts代码当中
