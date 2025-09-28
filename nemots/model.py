import time
from collections import defaultdict
from collections import deque

import numpy as np

from . import score
from . import symbolics
from .mcts import MCTS
from .network import PVNetCtx


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
        
        # 初始化use_adapter属性
        try:
            from mcts_adapter import MCTSAdapter
            self.use_adapter = True
        except ImportError:
            self.use_adapter = False

        # 自动推断n和lookBACK
        self.n = None
        self.lookBACK = getattr(args, 'lookBACK', 20)
        # 仅NEMoTS自动生成多变量多步grammar
        if self.symbolic_lib == 'NEMoTS':
            # 尝试从args或数据推断n
            if hasattr(args, 'n_vars'):
                self.n = args.n_vars
            else:
                self.n = None  # 运行时再推断
            # 动态生成grammar（如n未知则首次run时再生成）
            def gen_multivar_grammar(n, lookBACK):
                terminals = [f'A->x{i}' for i in range(n*lookBACK)]
                ops = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)']
                return ops + terminals
            
            if self.n is not None:
                self.base_grammar = gen_multivar_grammar(self.n, self.lookBACK)
                
            else:
                self.base_grammar = []  # run时再生成
                
        else:
            self.base_grammar = symbolics.rule_map[self.symbolic_lib]
        # 支持外部注入共享网络
        if hasattr(args, 'shared_network') and args.shared_network is not None:
            self.p_v_net_ctx = args.shared_network
        else:
            self.p_v_net_ctx = PVNetCtx(self.base_grammar, self.num_transplant, self.device)
        self.nt_nodes = symbolics.ntn_map[self.symbolic_lib]
        self.score_with_est = score.score_with_est
        self.data_buffer = deque(maxlen=10240)

        self.aug_grammars_counter = defaultdict(lambda: 0)

    def run(self, X, y=None, inherited_tree=None):
        assert X.size(0) == 1
        
        # 动态生成base_grammar（防止初始化时n未知导致grammar为空）
        if self.symbolic_lib == 'NEMoTS' and (self.base_grammar is None or len(self.base_grammar) == 0):
            X_ = X.squeeze(0)
            
            if X_.ndim == 2:
                if X_.shape[0] == self.lookBACK:
                    n = X_.shape[1]
                else:
                    n = X_.shape[0]
                self.n = n
            elif X_.ndim == 1:
                raise ValueError(f"无法从 X_ shape {X_.shape} 推断 n，请检查输入数据 shape！")
            else:
                raise ValueError(f"未知的 X_ shape: {X_.shape}，无法推断 n！")
            def gen_multivar_grammar(n, lookBACK):
                terminals = [f'A->x{i}' for i in range(n*lookBACK)]
                ops = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C', 'A->log(A)', 'A->sqrt(A)']
                return ops + terminals
            self.base_grammar = gen_multivar_grammar(self.n, self.lookBACK)
            
            # 使用__init__中已初始化的self.p_v_net_ctx，不需要再次初始化
            
            # 导入适配器模块，解决MCTS与神经网络维度不匹配问题
            try:
                from .mcts_adapter import MCTSAdapter
                self.use_adapter = True
            except ImportError:
                self.use_adapter = False
            
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
                # 展平输入
            X_flat = X_.reshape(-1)
            time_idx = np.arange(X_flat.shape[0])
            input_data = np.vstack([time_idx, X_flat])
            if y is not None:
                y = y.squeeze(0)
                # 保证y shape和input一致
                supervision_data = np.vstack([
                    np.arange(X_flat.shape[0] + y.numel()),
                    np.concatenate([X_flat.ravel(), y.ravel()])
                ])
            else:
                supervision_data = np.vstack([
                    np.arange(X_flat.shape[0]),
                    X_flat
                ])
            
        else:
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

        all_times = []
        all_eqs = []
        test_scores = []

        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant

        for i_test in range(self.num_runs):
            best_solution = ('nothing', 0)

            exploration_rate = self.exploration_rate  # 设置探索率
            max_module = self.max_module_init  # 设置最大模块
            reward_his = []  # 初始化奖励历史
            best_modules = []  # 初始化最佳模块
            aug_grammars = []  # 初始化增强语法

            start_time = time.time()  # 记录开始时间

            self.p_v_net_ctx.reset_grammar_vocab_name()

            for i_itr in range(self.num_transplant):
                # 只在第一次迭代时传递inherited_tree
                current_inherited_tree = inherited_tree if i_itr == 0 else None
                
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
                                  inherited_tree=current_inherited_tree)
                
                # 使用适配器修补MCTS实例，解决维度不匹配问题
                if self.use_adapter:
                    from .mcts_adapter import MCTSAdapter
                    mcts_block = MCTSAdapter.patch_mcts(mcts_block, self.p_v_net_ctx)
                    print(f"[信息] MCTS适配器已应用，解决维度不匹配问题")
                

                # 判断经验池是否已满，决定是否用网络引导
                buffer_size = self.data_buffer.maxlen
                current_buffer_size = len(self.data_buffer)
                warmup = int(buffer_size * 0.1)
                if current_buffer_size < warmup:
                    alpha = 0.0
                else:
                    alpha = min(1.0, (current_buffer_size - warmup) / (buffer_size - warmup))
                _, current_solution, good_modules, records = mcts_block.run(
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

                self.data_buffer.extend(list(records)[:])

                # 如果没有最佳模块，则将好的模块赋值给最佳模块
                if not best_modules:
                    best_modules = good_modules
                else:
                    # 否则，将最佳模块和好的模块合并，并按照评分进行排序
                    best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

                # 更新增强语法
                aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]
                for grammar in aug_grammars:
                    self.aug_grammars_counter[grammar] += 1

                # 将最佳解决方案的评分添加到奖励历史中
                reward_his.append(best_solution[1])

                # 如果当前解决方案的评分大于最佳解决方案的评分，则更新最佳解决方案
                if current_solution[1] > best_solution[1]:
                    best_solution = current_solution

                # 增加最大模块
                max_module += module_grow_step
                # 增加探索率
                exploration_rate *= 5

                # 检查是否发现了解决方案。如果是，提前停止。
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

        # 返回 policy_batch[0] 和 reward_his[-1]
        policy = policy_batch[0] if 'policy_batch' in locals() and len(policy_batch) > 0 else None
        reward = reward_his[-1] if len(reward_his) > 0 else None
        return all_eqs, all_times, test_scores, supervision_data, policy, reward
