
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
        self.data_buffer = deque(maxlen=10240)
        #nt_nodes: 获取非终端节点（Non-terminal nodes），在语法树中通常代表运算操作。
        # `score_with_est`: 指定用于评估表达式好坏的评分函数*，该函数来自 score.py。
    #  *   data_buffer: 创建一个固定长度的队列，用作经验回放池。

        self.aug_grammars_counter = defaultdict(lambda: 0)


#！执行符号回归任务的入口，包含了算法的核心循环
    def run(self, X, y=None, previous_best_tree=None):
        #通过这里的操作，外部接口，由engine中的simulate方法调用
        # assert X.size(0) == 1 # 注释掉此行，以允许处理来自滑动窗口的批数据
        #一个断言，确保每次只处理一个数据样本（batch_size 为 1）。这是因为 MCTS搜索是针对单个样本进行的
        
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
#初始化三个列表，用于存放多次运行 (num_runs) 的结果：耗时、找到的表达式和它们的分数

        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant
 #计算模块的步长
        for i_test in range(self.num_runs):
            #开始外部循环，执行 num_runs 次独立的、完整的符号回归搜索，以提高找到最优解的概率。
            best_solution = ('nothing', 0)

            exploration_rate = self.exploration_rate  # 设置探索率
            max_module = self.max_module_init  # 设置最大模块
            reward_his = []  # 初始化奖励历史
            best_modules = []  # 初始化最佳模块
            aug_grammars = []  # 初始化增强语法

            start_time = time.time()  # 记录开始时间

            self.p_v_net_ctx.reset_grammar_vocab_name()

            for i_itr in range(self.num_transplant):
                # 开始内部循环，这是算法的核心，被称为“移植（transplant）”或迭代增强过程。
                mcts_block = MCTS(data_sample=supervision_data,
                                  base_grammars=self.base_grammar,
                                  aug_grammars=[x[0] for x in sorted(self.aug_grammars_counter.items(), key=lambda item: item[1], reverse=True)[:20]],
                                  nt_nodes=self.nt_nodes,
                                  max_len=self.max_len,
                                  max_module=max_module,
                                  aug_grammars_allowed=self.num_aug,
                                  func_score=self.score_with_est,
                                  exploration_rate=self.exploration_rate,
                                  eta=self.eta,
                                  initial_tree=previous_best_tree)
                #实例化 MCTS 引擎*。
     #接口*: 这是与 mcts.py 模块最核心的接口。它创建了 MCTS 类的一个实例，并传入了大量配置参数：
      #*   data_sample: 用于评分的数据。
      #*   base_grammars, aug_grammars: 基础语法和从上一轮学到的增强语法。
      #*   nt_nodes: 非终端节点，来自 symbolics.py 模块。
       #  `func_score=self.score_with_est`: 函数回调接口*。将 score.py 模块中的评分函数 score_with_est 作为一个参数传递给 MCTS，MCTS内部会调用这个函数来评估它发现的表达式的好坏
                #
                #从mcts导入了MCTS 这里把MCTS模式改变后，对应的参数肯定变化，这里肯定要动刀


                

                # 判断经验池是否已满，决定是否用网络引导  这一部分不用修改
                buffer_size = self.data_buffer.maxlen
                current_buffer_size = len(self.data_buffer)
                warmup = int(buffer_size * 0.1)
                if current_buffer_size < warmup:
                    alpha = 0.0
                else:
                    alpha = min(1.0, (current_buffer_size - warmup) / (buffer_size - warmup))
                    #计算alpha值，用于warmup，
                    # 这部分逻辑用于实现一个“预热”（warmup）阶段。在经验池 data_buffer 中的数据量较少时，alpha 值较低，MCTS
  #更依赖随机搜索；随着数据增多，alpha 增大，MCTS 会更依赖神经网络的引导

                #————执行MCTS搜索———— 调用的是mcts文件中的run部分
                new_best_tree_node, current_solution, good_modules, records = mcts_block.run(
                    input_data,
                    self.transplant_step,
                    network=self.p_v_net_ctx,
                    num_play=10,
                    print_flag=True,
                    use_network=True,
                    alpha=alpha,
                    buffer_size=buffer_size,
                    current_buffer_size=current_buffer_size
                )
                #按照mcts中核心run方式格式进行，具体细节看mcts 这里要大改
#这是与mcts接口的调用 它执行了 mcts.py 中 MCTS 实例的 run 方法
                #network=self.p_v_net_ctx`: 核心接口*。将 network.py 中定义的神经网络对象 p_v_net_ctx 传递给 MCTS。在 MCTS 内部，它会反复调用network.forward() 方法来获取策略（下一步该如何构建表达式）和价值（当前表达式的潜力评分），从而引导整个搜索过程。
# use_network=True: 明确指示 MCTS 使用神经网络进行引导。
  # 返回值*:
      #*   current_solution: 本轮 MCTS 找到的最佳表达式及其分数。
      #*   good_modules: 搜索过程中发现的有潜力的子表达式（模块）。
      #*   records: 搜索过程的详细记录（(状态, 策略, 价值)元组），用于填充经验回放池

                self.data_buffer.extend(list(records)[:])
                #将 MCTS返回的 records添加到经验回放池self.data_buffe 中。这些数据未来可用于重新训练神经网络，使其引导能力越来越强
                # 更新最佳模块 如果没有最佳模块，则将好的模块赋值给最佳模块
                if not best_modules:
                    best_modules = good_modules
                else:
                    # 否则，将最佳模块和好的模块合并，并按照评分进行排序
                    best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

                # 更新增强语法
                aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]
                for grammar in aug_grammars:
                    self.aug_grammars_counter[grammar] += 1

                # 将最佳解决方案的评分添加到奖励历史中  这种历史概念是不是要用到我们的改造当中
                reward_his.append(best_solution[1])

                # 如果当前解决方案的评分大于最佳解决方案的评分，则更新最佳解决方案
                if current_solution[1] > best_solution[1]:
                    best_solution = current_solution
                    final_best_tree = new_best_tree_node

                # 增加最大模块 这里的模块对应的概念是有用的子表达式，例如符号增强吧
                max_module += module_grow_step
                # 增加探索率
                exploration_rate *= 5

                # 检查是否发现了解决方案。如果是，提前停止。
                #在一轮内部迭代结束后，对当前找到的最佳解进行一次最终评分。
                test_score = \
                    self.score_with_est(score.simplify_eq(best_solution[0]), 0, supervision_data, eta=self.eta)[0]
#接口*:
      #*   score.simplify_eq(): 调用 score.py (或相关工具模块) 中的函数来化简表达式字符串。
      #*   self.score_with_est(): 再次调用来自 score.py 的评分函数来计算最终分数。

            all_eqs.append(score.simplify_eq(best_solution[0]))
            test_scores.append(test_score)

            eq_out = score.simplify_eq(best_solution[0])
            if self.symbolic_lib == 'NEMoTS':
                print(f'变量映射: {[f"x{i}" for i in range(self.n*self.lookBACK)]}')
            print('\n{} tests complete after {} iterations.'.format(i_test + 1, i_itr + 1))
            print('best solution: {}'.format(eq_out))
            print('test score: {}'.format(test_score))
            print()

        # 返回 policy_batch[0] 和 reward_his[-1]
        policy = None # Placeholder, policy_batch is not available in this scope
        reward = reward_his[-1] if len(reward_his) > 0 else None
        return all_eqs, all_times, test_scores, supervision_data, policy, reward, final_best_tree


    #module就是语法增强最直接的利用对象：子结构  会被添加到语法增加库当中 #细节还是在mcts代码当中
