import math
import math
import random
from collections import defaultdict
       #！！！！！ engine是一个大框架，很多都是接的model模块 进行改造时，稍稍变动即可
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from torch.distributions import Categorical

from .model import Model
#这是engine最重要的组件，model类封装了,MCTS搜索和策略价值网络的组合逻辑
from .tracker import Tracker
#用于在训练过程中记录和可视化各项指标


class Engine(object):
    def __init__(self, args):
        self.args = args
        # args 对象通常是用来接收和管理从命令行传入脚本的参数的。在 Python 脚本中，这通常是通过 argparse
  #模块实现的。它将命令行参数（例如，python main.py --batch_size 32 --learning_rate
  #0.001）解析到一个对象中，这个对象通常被命名为 args。然后，代码就可以通过 args.batch_size 或
  #args.learning_rate 这样的方式来访问这些参数值
        #将参数对象保存为类的成员变量，方便后续方法使用
        self.model = Model(args)
        #模块互通，穿件核心的model类的实例，最重要的交互点，engine创建并持有了model对象
        #model对象内部会进一步创建MCTS实例和神经网络实例
        self.model.p_v_net_ctx.pv_net = self.model.p_v_net_ctx.pv_net.to(self.args.device)
        #将模型内部的策略价值网络移动到指定的设备上
        self.optimizer = op.Adam(self.model.p_v_net_ctx.pv_net.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)
        #创建Adam优化器 该优化器需要优化的参数来源于self.model.p_v_net_ctx.pv_net.parameters()
        #这表明engine通过model对象，直接访问并管理其内部神经网络的参数
        self.tracker = Tracker()
        self.global_train_step = 0
        #初始化一个全局训练不属计数器

        #————模拟方法————
        #{接口} 此方法被main.py中的训练循环直接调用
    def simulate(self, data, previous_best_tree=None):
        # 兼容处理Numpy数组和PyTorch张量
        if isinstance(data, torch.Tensor):
            # 如果是批处理数据，先移除批次维度
            if data.dim() == 3 and data.shape[0] == 1:
                data = data.squeeze(0)
            data = data.cpu().numpy()

        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        #将一份传入的data切分为输入序列和目标序列
        # 将numpy数组转换为PyTorch张量，因为模型内部期望张量对象
        X = torch.from_numpy(X).float().to(self.args.device)
        y = torch.from_numpy(y).float().to(self.args.device)
        all_eqs, all_times, test_scores, test_data, policy, reward, new_best_tree = self.model.run(X, y, previous_best_tree=previous_best_tree)
        #{接口}engine将数据X,y直接交给model，并由model内部完成MCTS搜索、与神经网络交互、填充经验池等一系列操作 并不关心据图细节，交给model，只负责调用和接受结果
        mae, mse, corr, best_exp = OptimizedMetrics.metrics(all_eqs, test_scores, test_data)
        #{接口} 调用本文件内定义的OptimizedMetrics类的静态方法metrics。
        #用于从MCTS返回的多个表达式中，根据分数选出最佳表达式，并计算其MAE,MSE,CORR指标

        # 【模块互通】检查模型内部数据缓冲区(data_buffer)的大小。

        #这个缓冲区是在self.model.run方法中被填充的，这里Engine读取其状态，这是一种通过共享状态实现的模块间通信
        if len(self.model.data_buffer) > self.args.train_size:
            # 如果经验池中的数据足够多，就调用自身的train方法来训练神经网络。
            loss = self.train()
            return best_exp, all_times, test_data, loss, mae, mse, corr, policy, reward, new_best_tree
        # 返回包括训练损失在内的所有结果给main.py。
        return best_exp, all_times, test_data, 0, mae, mse, corr, policy, reward, new_best_tree
        # # 如果经验池数据不足，则不进行训练，损失计为0。


        #————训练方法————
    def train(self):
        self.model.p_v_net_ctx.pv_net.train()
        print("start train neural networks...")
        cumulative_loss = 0
        
        # 从经验池中一次性采样全量数据
        state_batch_full, seq_batch_full, policy_batch_full, value_batch_full, reward_batch_full, _ = self.preprocess_data()

        if not state_batch_full:
            print("[WARN] No data sampled from memory buffer for training.")
            return 0

        mini_batch_size = 64  # 定义一个安全的迷你批次大小

        for epoch in range(self.args.epoch):
            # 每个epoch开始时，都打乱一次数据顺序
            indices = list(range(len(state_batch_full)))
            random.shuffle(indices)
            
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(state_batch_full), mini_batch_size):
                self.optimizer.zero_grad()

                # 2. 获取当前迷你批次的索引
                mini_batch_indices = indices[i:i + mini_batch_size]

                # 3. 根据索引创建迷你批次
                state_batch = [state_batch_full[j] for j in mini_batch_indices]
                seq_batch = [seq_batch_full[j] for j in mini_batch_indices]
                policy_batch = [policy_batch_full[j] for j in mini_batch_indices]
                value_batch = torch.Tensor([value_batch_full[j] for j in mini_batch_indices])
                reward_batch = torch.Tensor([reward_batch_full[j] for j in mini_batch_indices])
                
                length_indices = self.obtain_policy_length(policy_batch)

                print(f"[DEBUG] Epoch {epoch+1}, Mini-batch {i//mini_batch_size + 1}, Size: {len(state_batch)}")
                if len(state_batch) == 0 or len(seq_batch) == 0:
                    print("[WARN] Skipped an empty mini-batch.")
                    continue

                # 【模块互通】对迷你批次进行前向传播
                raw_dis_out, value_out, reward_out = self.model.p_v_net_ctx.policy_value_batch(seq_batch, state_batch)

                # --- 计算损失 ---
                value_batch[torch.isnan(value_batch)] = 0.
                value_loss = F.mse_loss(value_out, value_batch.to(value_out.device))
                
                # 新增：reward损失计算
                reward_batch[torch.isnan(reward_batch)] = 0.
                reward_loss = F.mse_loss(reward_out, reward_batch.to(reward_out.device))

                dist_loss = []
                if length_indices: # 确保length_indices不为空
                    for length, sample_id in length_indices.items():
                        if not sample_id: continue
                        out_policy = F.softmax(torch.stack([raw_dis_out[k] for k in sample_id])[:, :length], dim=-1)
                        gt_policy = torch.Tensor([policy_batch[k] for k in sample_id]).to(out_policy.device)
                        dist_target = Categorical(probs=gt_policy)
                        dist_out = Categorical(probs=out_policy)
                        dist_loss.append(torch.distributions.kl_divergence(dist_target, dist_out).mean())
                
                # 总损失 = 价值损失 + 策略损失 + reward损失
                if not dist_loss or not any(dist_loss):
                    total_loss = value_loss + reward_loss
                else:
                    total_loss = value_loss + sum(dist_loss) + reward_loss
                
                epoch_loss += total_loss.item()
                num_batches += 1

                # --- 反向传播与优化 ---
                total_loss.backward() # 使用 .backward() 即可，无需 retain_graph=True
                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
                
                self.optimizer.step()

                # --- 日志记录 ---
                alpha = min(1.0, len(self.model.data_buffer) / self.model.data_buffer.maxlen)
                policy = policy_batch[0] if len(policy_batch) > 0 else None
                value = value_batch.mean().item() if len(value_batch) > 0 else None
                
                self.tracker.update(
                    step=self.global_train_step,
                    alpha=alpha,
                    policy=policy,
                    value=value,
                    reward=None, # reward等指标在simulate阶段记录更合适
                    corr=None,
                    best_score=None
                )
                self.global_train_step += 1
            
            if num_batches > 0:
                cumulative_loss += (epoch_loss / num_batches)

        print("end train neural networks...")
        self.tracker.plot()
        self.tracker.save_npz()
        return cumulative_loss / self.args.epoch if self.args.epoch > 0 else 0
        print("end train neural networks...")
        self.tracker.plot()
        self.tracker.save_npz()
        return cumulative_loss / self.args.epoch
    # 返回本次训练的平均损失。

    # --- 4. 辅助与评估方法 ---
    # 辅助函数：根据策略列表的子列表长度进行分组，返回一个记录了长度和对应索引的字典。
    def obtain_policy_length(self, policy):
        length_indices = defaultdict(list)
        for idx, sublist in enumerate(policy):
            length_indices[len(sublist)].append(idx)
        return dict(length_indices)

    # 辅助函数：预处理数据，用于从经验池中采样并组织成批次。
    def preprocess_data(self):
        non_nan_indices = [index for index, value in enumerate(self.model.data_buffer) if not math.isnan(value[3])]
        # 【模块互通】从self.model.data_buffer（经验池）中进行采样。
        # 过滤掉值为NaN的样本。
        sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        mini_batch = [self.model.data_buffer[i] for i in sampled_idx]
        # 随机采样指定数量的样本。 通过train-size指定

        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        value_batch = [data[3] for data in mini_batch]
        # 新增：提取reward数据（假设在data[4]位置）
        reward_batch = [data[4] if len(data) > 4 else 0.0 for data in mini_batch]
        # 将采样出的数据解包成state, seq, policy, value, reward等不同的批次

        length_indices = self.obtain_policy_length(policy_batch)
        # 调用obtain_policy_length对策略进行分组。
        return state_batch, seq_batch, policy_batch, value_batch, reward_batch, length_indices


    def eval(self, data):
        pass
    # 评估方法，目前是空的，没有实现。 这里要添加评估方式 前面不是已经算出指标了 可以利用这

# 定义一个独立的指标计算类。
class OptimizedMetrics:
    @staticmethod
    # 定义为静态方法，可以直接通过类名调用，无需创建实例。
    def metrics(exps, scores, data):
        best_index = np.argmax(scores)
        best_exp = exps[best_index]

        # --- Create a safe evaluation context ---
        eval_vars = {
            "np": np, # Allow numpy functions like np.exp
        }
        num_var = data.shape[0] - 1
        for i in range(num_var):
            eval_vars[f'x{i}'] = data[i, :]
        gt = data[-1, :]

        # Replacing the lambdify function with the new lambda function
        # Add "np." prefix to functions for safe eval
        corrected_expression = best_exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin","np.sin").replace("sqrt", "np.sqrt").replace("log", "np.log")
        
        # 警告：使用eval()函数存在严重的安全风险和稳定性问题，它会执行任意字符串代码。
        # 在生产环境中应避免使用，或用更安全的AST（抽象语法树）解析等方法替代。
        try:
            prediction = eval(corrected_expression, {"__builtins__": None}, eval_vars)
            # Handle cases where the expression is a constant
            if not isinstance(prediction, np.ndarray) or prediction.shape != gt.shape:
                if isinstance(prediction, (int, float)):
                    prediction = np.repeat(prediction, gt.shape)
                else:
                    # Fallback for complex mismatches
                    prediction = np.zeros_like(gt)

        except Exception:
            # If eval fails for any reason, predict zeros
            prediction = np.zeros_like(gt)

        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)
        # 计算MAE（平均绝对误差）和MSE（均方误差）。
        corr = 0.0  # 增加默认值，防止异常分支下未赋值
        try:
            corr, _ = pearsonr(prediction, gt)
        except ValueError:
            if (np.isnan(prediction) | np.isinf(prediction)).any():
                corr = 0.
            elif (np.isnan(gt) | np.isinf(gt)).any():
                valid_indices = np.where(~np.isnan(gt) & ~np.isinf(gt))[0]
                valid_gt = gt[valid_indices]
                valid_pred = prediction[valid_indices]
                corr, _ = pearsonr(valid_pred, valid_gt)
        except TypeError:
            if type(prediction) is float:
                corr = 0.

        return mae, mse, corr, best_exp

    def train_with_quantile_loss(self, predictions, targets):
        """
        使用分位数损失训练PVNET
        
        Args:
            predictions: 预测值数组 [num_samples, seq_len]
            targets: 真实值数组 [seq_len]
            
        Returns:
            dict: 训练指标
        """
        # 计算分位数损失
        quantile_loss = self.model.p_v_net_ctx.pv_net.compute_quantile_loss(predictions, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        quantile_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
        
        # 更新参数
        self.optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            if isinstance(predictions, torch.Tensor):
                pred_np = predictions.cpu().numpy()
            else:
                pred_np = np.array(predictions)
            
            if isinstance(targets, torch.Tensor):
                target_np = targets.cpu().numpy()
            else:
                target_np = np.array(targets)
            
            # 计算Q25和Q75
            if pred_np.ndim == 2 and pred_np.shape[0] > 1:
                q25 = np.percentile(pred_np, 25, axis=0)
                q75 = np.percentile(pred_np, 75, axis=0)
            else:
                q25 = pred_np.flatten()
                q75 = pred_np.flatten()
            
            # 计算覆盖率
            coverage_25 = np.mean(target_np >= q25)
            coverage_75 = np.mean(target_np <= q75)
            coverage_both = np.mean((target_np >= q25) & (target_np <= q75))
        
        return {
            'quantile_loss': quantile_loss.item(),
            'q25_values': q25,
            'q75_values': q75,
            'coverage_25': coverage_25,
            'coverage_75': coverage_75,
            'coverage_both': coverage_both
        }

# Example usage (assuming exps, scores, and data are defined)
# metrics = OptimizedMetrics.metrics(exps, scores, data)
#Engine 类是连接高层控制 (main.py) 和底层算法实现 (model.py, network.py, mcts.py) 的核心粘合剂。