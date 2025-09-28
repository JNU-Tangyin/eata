import math
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from torch.distributions import Categorical

from .model import Model
from .tracker import Tracker


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
        self.model.p_v_net_ctx.pv_net = self.model.p_v_net_ctx.pv_net.to(self.args.device)
        self.optimizer = op.Adam(self.model.p_v_net_ctx.pv_net.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)
        self.tracker = Tracker()
        self.global_train_step = 0

    def simulate(self, data, inherited_tree=None):
        X, y = data[:, :self.args.seq_in], data[:, -self.args.seq_out:]
        all_eqs, all_times, test_scores, test_data, policy, reward = self.model.run(X, y, inherited_tree=inherited_tree)
        print(f"🔍 调用OptimizedMetrics.metrics:")
        print(f"   all_eqs: {all_eqs}")
        print(f"   test_scores: {test_scores}")
        print(f"   test_data类型: {type(test_data)}")
        if hasattr(test_data, 'shape'):
            print(f"   test_data形状: {test_data.shape}")
        else:
            print(f"   test_data内容: {test_data}")
        
        mae, mse, corr, best_exp = OptimizedMetrics.metrics(all_eqs, test_scores, test_data)
        
        print(f"🔍 OptimizedMetrics.metrics返回:")
        print(f"   mae: {mae}, mse: {mse}, corr: {corr}")
        print(f"   best_exp: {best_exp}")
        
        # 保存最近的训练结果供tracker使用
        self._last_reward = reward if reward is not None else 0.0
        self._last_corr = corr if corr is not None else 0.0
        self._last_best_score = max(test_scores) if test_scores and len(test_scores) > 0 else 0.0
        
        if len(self.model.data_buffer) > self.args.train_size:
            loss = self.train()
            return best_exp, all_times, test_data, loss, mae, mse, corr, policy, reward
        return best_exp, all_times, test_data, 0, mae, mse, corr, policy, reward

    def train(self):
        self.model.p_v_net_ctx.pv_net.train()
        print("start train neural networks...")
        cumulative_loss = 0
        for epoch in range(self.args.epoch):
            self.optimizer.zero_grad()
            state_batch, seq_batch, policy_batch, value_batch, length_indices = self.preprocess_data()
            value_batch = torch.Tensor(value_batch)
            print(f"[DEBUG] state_batch len: {len(state_batch)}, seq_batch len: {len(seq_batch)}")
            if len(state_batch) == 0 or len(seq_batch) == 0:
                raise ValueError("[DEBUG] state_batch或seq_batch为空，请检查数据采样或经验池填充逻辑！")
            raw_dis_out, value_out = self.model.p_v_net_ctx.policy_value_batch(seq_batch, state_batch)
            value_batch[torch.isnan(value_batch)] = 0.
            value_loss = F.mse_loss(value_out, value_batch.to(value_out.device))
            dist_loss = []
            for length, sample_id in length_indices.items():
                try:
                    # 检查sample_id是否为空
                    if len(sample_id) == 0:
                        continue
                    
                    # 检查policy_batch中的数据
                    valid_policies = []
                    valid_raw_outs = []
                    for i in sample_id:
                        if i < len(policy_batch) and i < len(raw_dis_out):
                            policy = policy_batch[i]
                            if policy is not None and len(policy) > 0:
                                valid_policies.append(policy)
                                valid_raw_outs.append(raw_dis_out[i])
                    
                    if len(valid_policies) == 0:
                        continue
                        
                    out_policy = F.softmax(torch.stack(valid_raw_outs)[:, :length], dim=-1)
                    gt_policy = torch.Tensor(valid_policies).to(out_policy.device)
                    
                    # 确保形状匹配
                    if gt_policy.shape != out_policy.shape:
                        min_len = min(gt_policy.shape[1], out_policy.shape[1])
                        gt_policy = gt_policy[:, :min_len]
                        out_policy = out_policy[:, :min_len]
                    
                    dist_target = Categorical(probs=gt_policy)
                    dist_out = Categorical(probs=out_policy)
                    dist_loss.append(torch.distributions.kl_divergence(dist_target, dist_out).mean())
                    
                except Exception as e:
                    print(f"策略分布计算错误 (length={length}): {e}")
                    continue
            total_loss = value_loss + sum(dist_loss)
            cumulative_loss += total_loss.item()
            total_loss.backward(retain_graph=True)
            if self.args.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
            self.optimizer.step()

            # ===== 采集tracker指标 =====
            # alpha、policy_entropy、policy_maxprob、value、reward、corr、best_score、train_step
            # alpha由model.data_buffer长度决定
            alpha = min(1.0, len(self.model.data_buffer) / self.model.data_buffer.maxlen)
            # policy分布用一个batch的第一个policy为代表
            if len(policy_batch) > 0:
                policy = policy_batch[0]
            else:
                policy = None
            # value用当前batch均值
            value = value_batch.mean().item() if len(value_batch) > 0 else None
            # reward、corr、best_score用最近一次simulate的结果
            # 从实例变量中获取最近的训练结果
            reward = getattr(self, '_last_reward', 0.0)
            corr = getattr(self, '_last_corr', 0.0) 
            best_score = getattr(self, '_last_best_score', 0.0)
            self.tracker.update(
                step=self.global_train_step,
                alpha=alpha,
                policy=policy,
                value=value,
                reward=reward,
                corr=corr,
                best_score=best_score
            )
            self.global_train_step += 1
        print("end train neural networks...")
        self.tracker.plot()
        self.tracker.save_npz()
        return cumulative_loss / self.args.epoch

    def obtain_policy_length(self, policy):
        length_indices = defaultdict(list)
        for idx, sublist in enumerate(policy):
            length_indices[len(sublist)].append(idx)
        return dict(length_indices)

    def preprocess_data(self):
        non_nan_indices = [index for index, value in enumerate(self.model.data_buffer) if not math.isnan(value[3])]
        sampled_idx = random.sample(non_nan_indices, min(len(non_nan_indices), self.args.train_size))
        mini_batch = [self.model.data_buffer[i] for i in sampled_idx]
        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        value_batch = [data[3] for data in mini_batch]
        length_indices = self.obtain_policy_length(policy_batch)
        return state_batch, seq_batch, policy_batch, value_batch, length_indices

    def eval(self, data):
        pass


class OptimizedMetrics:
    @staticmethod
    def metrics(exps, scores, data):
        # 修复tuple index out of range错误
        if len(scores) == 0 or len(exps) == 0:
            return 0.0, 0.0, 0.0, "x0"
        
        best_index = np.argmax(scores)
        if best_index >= len(exps):
            best_index = 0
        best_exp = exps[best_index]
        
        # 安全地解包data
        try:
            if isinstance(data, tuple) and len(data) >= 2:
                span, gt = data[0], data[1]
            elif isinstance(data, np.ndarray) and data.shape[0] >= 2:
                # 如果是numpy数组，第一行是输入，第二行是目标
                span, gt = data[0], data[1]
                print(f"🔍 数据解包成功: span={span}, gt={gt}")
            else:
                # 如果data格式不对，返回默认值
                print(f"🔍 数据格式不支持: {type(data)}, shape={getattr(data, 'shape', 'N/A')}")
                return 0.0, 0.0, 0.0, str(best_exp)
        except (ValueError, IndexError) as e:
            print(f"数据解包错误: {e}")
            return 0.0, 0.0, 0.0, str(best_exp)

        # 确保span和gt是numpy数组且形状匹配
        try:
            span = np.asarray(span)
            gt = np.asarray(gt)
            
            if span.shape != gt.shape:
                min_len = min(len(span), len(gt))
                span = span[:min_len]
                gt = gt[:min_len]
        except Exception as e:
            print(f"数组处理错误: {e}")
            return 0.0, 0.0, 0.0, str(best_exp)

        # Replacing the lambdify function with the new lambda function
        try:
            corrected_expression = str(best_exp).replace("exp", "np.exp").replace("cos", "np.cos").replace("sin",
                                                                                                      "np.sin").replace(
                "sqrt", "np.sqrt").replace("log", "np.log")
            
            print(f"🔍 评估表达式: {corrected_expression}")
            print(f"   span形状: {span.shape if hasattr(span, 'shape') else type(span)}")
            print(f"   gt形状: {gt.shape if hasattr(gt, 'shape') else type(gt)}")
            
            # 设置变量x0, x1, x2等供表达式使用
            for i in range(len(span)):
                globals()[f'x{i}'] = span[i]
            
            # 如果表达式中包含x0但span长度不够，使用span[0]
            if 'x0' in corrected_expression and len(span) > 0:
                globals()['x0'] = span[0]
            
            f = lambda x: eval(corrected_expression)
            prediction = f(span)
            
            print(f"   预测结果: {prediction}")
            print(f"   真实值: {gt}")
            
            # 确保prediction是数组且形状正确
            prediction = np.asarray(prediction)
            if prediction.shape != gt.shape:
                if prediction.size == 1:
                    prediction = np.full_like(gt, prediction.item())
                else:
                    min_len = min(len(prediction), len(gt))
                    prediction = prediction[:min_len]
                    gt = gt[:min_len]
                    
        except Exception as e:
            print(f"表达式评估错误: {e}")
            return 0.0, 0.0, 0.0, str(best_exp)

        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)
        corr = 0.0  # 增加默认值，防止异常分支下未赋值
        
        try:
            # 确保prediction和gt都是1维数组且长度相同
            pred_flat = prediction.flatten()
            gt_flat = gt.flatten()
            
            if len(pred_flat) != len(gt_flat):
                min_len = min(len(pred_flat), len(gt_flat))
                pred_flat = pred_flat[:min_len]
                gt_flat = gt_flat[:min_len]
            
            if len(pred_flat) > 1:  # pearsonr需要至少2个数据点
                corr, _ = pearsonr(pred_flat, gt_flat)
            else:
                corr = 0.0
                
        except (ValueError, IndexError) as e:
            print(f"相关性计算错误: {e}")
            if (np.isnan(prediction) | np.isinf(prediction)).any():
                corr = 0.
            elif (np.isnan(gt) | np.isinf(gt)).any():
                try:
                    valid_indices = np.where(~np.isnan(gt) & ~np.isinf(gt))[0]
                    if len(valid_indices) > 1:
                        valid_gt = gt[valid_indices]
                        valid_pred = prediction[valid_indices]
                        corr, _ = pearsonr(valid_pred, valid_gt)
                    else:
                        corr = 0.0
                except:
                    corr = 0.0
        except TypeError:
            if type(prediction) is float:
                corr = 0.

        return mae, mse, corr, best_exp

# Example usage (assuming exps, scores, and data are defined)
# metrics = OptimizedMetrics.metrics(exps, scores, data)
