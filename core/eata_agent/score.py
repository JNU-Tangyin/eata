import _thread
import threading
from contextlib import contextmanager

import numpy as np
from numpy import *
from gplearn.functions import make_function
from scipy.optimize import minimize
from sympy import simplify, expand


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    if eq is None or not isinstance(eq, str) or eq.strip() == "":
        return "0"
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999):
    if not eq or not isinstance(eq, str) or not eq.strip():
        return 0, "0"

    """
    该函数计算一个完整解析树的奖励分数。
    如果方程中包含占位符C，也会为C执行估计。
    奖励 = 1 / (1 + MSE) * Penalty ** num_term

    这是主函数的开始，它接受五个参数：eq（一个字符串，表示解析树生成的方程式），tree_size（整数，解析树的大小），
    data（二维 numpy 数组，表示用于评分的数据），t_limit（表示计算评分的时间限制）和 eta（一个用于计算惩罚因子的超参数）。

    参数:
    eq : 字符串对象，已发现的方程（包含占位符C的系数）。
    tree_size : 整数对象，完整解析树中的产生规则数。
    data : 二维numpy数组，测量数据，包括独立变量和因变量（最后一行）。
    t_limit : 浮点数对象，单次评估的时间限制（秒），默认为1秒。

    返回值:
    score: 浮点数，已发现的方程的奖励分数。
    eq: 字符串，包含估计数值的已发现方程。
    """
    
    local_vars = {
        'cos': np.cos,
        'sin': np.sin,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
    }

    try:
        num_var = data.shape[0] - 1
        for i in range(num_var):
            local_vars[f'x{i}'] = data[i, :]
        f_true = data[-1, :]
        local_vars['f_true'] = f_true

        c_count = eq.count('C')

        with time_limit(t_limit, 'evaluate_equation'):
            f_pred = np.array([]) # Initialize f_pred
            if c_count == 0:
                f_pred = eval(eq, {"__builtins__": None}, local_vars)
            elif c_count >= 10:
                return 0, eq
            else:
                c_lst_names = ['c' + str(i) for i in range(c_count)]
                eq_with_c_vars = eq
                for c_name in c_lst_names:
                    eq_with_c_vars = eq_with_c_vars.replace('C', c_name, 1)

                def eq_test(c_values):
                    # Create a temporary context for optimization
                    opt_vars = local_vars.copy()
                    for i in range(len(c_values)):
                        opt_vars[c_lst_names[i]] = c_values[i]
                    
                    try:
                        pred = eval(eq_with_c_vars, {"__builtins__": None}, opt_vars)
                        # Ensure pred is a numpy array
                        if not isinstance(pred, np.ndarray):
                            pred = np.repeat(pred, len(f_true))
                        
                        # Handle shape mismatch
                        if pred.shape != f_true.shape:
                            # This might happen if the expression is a constant
                            if pred.size == 1:
                                pred = np.repeat(pred, f_true.shape)
                            else: # More complex mismatch, return high error
                                return np.inf
                        
                        return np.linalg.norm(pred - f_true, 2)
                    except Exception:
                        return np.inf # Return a large error if eval fails

                x0 = [1.0] * len(c_lst_names)
                opt_result = minimize(eq_test, x0, method='Powell', tol=1e-6)
                c_lst_values = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in opt_result.x]
                
                eq_est = eq_with_c_vars
                final_eval_vars = local_vars.copy()
                for i in range(len(c_lst_values)):
                    eq_est = eq_est.replace(c_lst_names[i], str(c_lst_values[i]), 1)
                    final_eval_vars[c_lst_names[i]] = c_lst_values[i]

                eq = eq_est.replace('+-', '-')
                f_pred = eval(eq, {"__builtins__": None}, final_eval_vars)

        # Ensure f_pred is a numpy array of the correct shape
        if not isinstance(f_pred, np.ndarray) or f_pred.shape != f_true.shape:
             # Handle constants or shape mismatches
            if isinstance(f_pred, (int, float)):
                f_pred = np.repeat(f_pred, f_true.shape)
            else:
                # If it's still not matching, we can't calculate a meaningful score
                return 0, eq
        
        # Check for non-numeric or empty arrays
        if f_pred.size == 0 or not np.isfinite(f_pred).all():
            return 0, eq

        # Ensure f_pred and f_true are finite before calculating MSE
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=0.0, neginf=0.0)
        f_true = np.nan_to_num(f_true, nan=0.0, posinf=0.0, neginf=0.0)

        mse = np.linalg.norm(f_pred - f_true, 2)**2 / f_true.shape[0]
        # Temporarily simplify the reward to focus on MSE, ignoring complexity penalty
        r = 1.0 / (1.0 + mse)

    except Exception as e:
        # Any exception in this whole process means scoring fails
        # print(f"DEBUG: Scoring failed for eq='{eq}'. Error: {e}") # Optional: for debugging
        return 0, eq

    return r, eq


def simple_mae_score(eq, length, supervision_data, eta=0.999):
    """简单的MAE评分函数，用于EATA-Simple变体"""
    try:
        X, y = supervision_data
        f_true = y.flatten()
        
        # 评估表达式
        f_pred = eval(eq, {"__builtins__": {}}, dict(zip([f'x{i}' for i in range(X.shape[1])], X.T)))
        
        # 处理标量结果
        if isinstance(f_pred, (int, float)):
            f_pred = np.repeat(f_pred, f_true.shape)
        else:
            f_pred = np.array(f_pred).flatten()
        
        # 确保形状匹配
        if f_pred.shape != f_true.shape:
            if f_pred.size == 1:
                f_pred = np.repeat(f_pred[0], f_true.shape)
            else:
                return 0, eq
        
        # 检查有效性
        if f_pred.size == 0 or not np.isfinite(f_pred).all():
            return 0, eq
        
        # 处理无穷值和NaN
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=0.0, neginf=0.0)
        f_true = np.nan_to_num(f_true, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算MAE而不是MSE
        mae = np.mean(np.abs(f_pred - f_true))
        r = 1.0 / (1.0 + mae)
        
    except Exception as e:
        return 0, eq
    
    return r, eq


def score_with_objective(eq, length, data, t_limit=1.0, objective='mse', eta=0.999):
    """
    支持多种目标函数的评分函数，用于EATA-Simple变体
    
    参数:
        eq: 表达式字符串
        length: 表达式长度
        data: 监督数据，格式与score_with_est相同 (n_vars+1, n_samples)，最后一行是目标值
        t_limit: 时间限制（秒），与score_with_est保持一致（本函数暂不使用）
        objective: 目标函数类型 ('mse', 'kl', 'js', 'cvar')
        eta: 惩罚因子
    
    返回:
        (score, eq): 评分和表达式
    """
    # 检查表达式是否有效
    if not eq or not isinstance(eq, str) or not eq.strip():
        return 0, "0"
    
    try:
        # 使用与score_with_est相同的数据格式
        num_var = data.shape[0] - 1
        f_true = data[-1, :]
        
        # 评估表达式
        local_vars = {
            'cos': np.cos,
            'sin': np.sin,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
        }
        for i in range(num_var):
            local_vars[f'x{i}'] = data[i, :]
        
        f_pred = eval(eq, {"__builtins__": None}, local_vars)
        
        # 处理标量结果
        if isinstance(f_pred, (int, float)):
            f_pred = np.repeat(f_pred, len(f_true))
        else:
            f_pred = np.array(f_pred).flatten()
        
        # 确保形状匹配
        if f_pred.shape != f_true.shape:
            if f_pred.size == 1:
                f_pred = np.repeat(f_pred[0], len(f_true))
            else:
                return 0, eq
        
        # 检查有效性
        if f_pred.size == 0 or not np.isfinite(f_pred).all():
            return 0, eq
        
        # 处理无穷值和NaN
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=0.0, neginf=0.0)
        f_true = np.nan_to_num(f_true, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 根据目标函数计算距离
        if objective == 'mse':
            # MSE距离
            distance = np.mean((f_pred - f_true) ** 2)
        elif objective == 'kl':
            # KL散度：使用对称版本 0.5 * [KL(P||Q) + KL(Q||P)]，更稳定
            # 将预测和真实值归一化为概率分布
            pred_sum = np.sum(np.abs(f_pred))
            true_sum = np.sum(np.abs(f_true))
            
            # 检查是否可以归一化
            if pred_sum < 1e-10 or true_sum < 1e-10:
                # 无法归一化，使用MSE作为fallback
                distance = np.mean((f_pred - f_true) ** 2)
            else:
                pred_prob = np.abs(f_pred) / pred_sum
                true_prob = np.abs(f_true) / true_sum
                pred_prob = np.clip(pred_prob, 1e-10, 1.0)
                true_prob = np.clip(true_prob, 1e-10, 1.0)
                # 对称KL散度
                kl_pt = np.sum(pred_prob * np.log(pred_prob / true_prob))
                kl_tp = np.sum(true_prob * np.log(true_prob / pred_prob))
                distance = 0.5 * (kl_pt + kl_tp)
        elif objective == 'js':
            # JS散度
            pred_sum = np.sum(np.abs(f_pred))
            true_sum = np.sum(np.abs(f_true))
            
            # 检查是否可以归一化
            if pred_sum < 1e-10 or true_sum < 1e-10:
                # 无法归一化，使用MSE作为fallback
                distance = np.mean((f_pred - f_true) ** 2)
            else:
                pred_prob = np.abs(f_pred) / pred_sum
                true_prob = np.abs(f_true) / true_sum
                pred_prob = np.clip(pred_prob, 1e-10, 1.0)
                true_prob = np.clip(true_prob, 1e-10, 1.0)
                m = 0.5 * (pred_prob + true_prob)
                kl1 = np.sum(true_prob * np.log(true_prob / m))
                kl2 = np.sum(pred_prob * np.log(pred_prob / m))
                distance = 0.5 * (kl1 + kl2)
        elif objective == 'cvar':
            # CVaR (条件风险价值)
            errors = np.abs(f_pred - f_true)
            alpha = 0.05  # 5%分位数
            var = np.quantile(errors, 1 - alpha)
            tail_errors = errors[errors >= var]
            # 检查是否有尾部误差
            if len(tail_errors) == 0:
                # 没有尾部误差，使用最大误差
                distance = np.max(errors)
            else:
                distance = np.mean(tail_errors)
        else:
            # 默认使用MSE
            distance = np.mean((f_pred - f_true) ** 2)
        
        # 确保distance是有限的
        if not np.isfinite(distance) or distance < 0:
            return 0, eq
        
        # 计算奖励
        r = 1.0 / (1.0 + distance)
        
    except Exception as e:
        return 0, eq
    
    return r, eq

