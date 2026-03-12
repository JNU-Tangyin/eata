"""
计算符号表达式的复杂度（AST节点数）
用于Pareto Frontier分析
"""
import re
from typing import Union

def count_ast_nodes(expression: str) -> int:
    """
    计算符号表达式的AST节点数
    
    Args:
        expression: 符号表达式字符串，如 "x0 + x1 * 2"
        
    Returns:
        AST节点总数
        
    Examples:
        "x0" -> 1 (1个变量)
        "x0 + x1" -> 3 (2个变量 + 1个加法)
        "x0 * 2 + x1" -> 5 (2个变量 + 1个常数 + 1个乘法 + 1个加法)
    """
    if not expression or expression == '0':
        return 1
    
    # 移除空格
    expr = expression.replace(' ', '')
    
    # 计数器
    node_count = 0
    
    # 1. 计数变量（x0, x1, x2, ...）
    variables = re.findall(r'x\d+', expr)
    node_count += len(variables)
    
    # 2. 计数常数（整数和浮点数）
    # 先移除变量，避免重复计数
    temp_expr = re.sub(r'x\d+', '', expr)
    # 匹配数字（包括负数和小数）
    constants = re.findall(r'-?\d+\.?\d*', temp_expr)
    node_count += len([c for c in constants if c and c != '-'])
    
    # 3. 计数运算符
    operators = ['+', '-', '*', '/', '**', 'sqrt', 'log', 'exp', 'sin', 'cos', 'abs']
    for op in operators:
        if op in ['sqrt', 'log', 'exp', 'sin', 'cos', 'abs']:
            # 函数运算符
            node_count += expr.count(op)
        else:
            # 二元运算符（需要排除负号）
            if op == '-':
                # 只计数作为减法的负号，不计数作为负数的负号
                count = expr.count(op)
                # 粗略估计：开头的负号和运算符后的负号不计入
                for prev_op in ['+', '*', '/', '(', '**']:
                    count -= expr.count(prev_op + op)
                if expr.startswith(op):
                    count -= 1
                node_count += max(0, count)
            else:
                node_count += expr.count(op)
    
    # 4. 计数括号对（每对括号算1个节点）
    node_count += expr.count('(')
    
    return max(1, node_count)  # 至少为1


def estimate_method_complexity(method_name: str) -> int:
    """
    为非符号方法估算固定复杂度值
    
    Args:
        method_name: 方法名称
        
    Returns:
        估算的复杂度值
    """
    complexity_map = {
        # 简单基线
        'buy_and_hold': 1,
        
        # 技术指标
        'macd': 8,
        'rsi': 6,
        'bollinger': 10,
        
        # 传统机器学习（基于模型参数数量的粗略估计）
        'arima': 15,
        'lightgbm': 100,
        'xgboost': 120,
        
        # 深度学习（基于网络层数和参数的粗略估计）
        'lstm': 500,
        'transformer': 800,
        'gru': 450,
        
        # 强化学习
        'finrl_ppo': 600,
        'finrl_a2c': 550,
        'finrl_sac': 650,
        'finrl_td3': 620,
        'finrl_ddpg': 580,
    }
    
    return complexity_map.get(method_name.lower(), 50)


if __name__ == '__main__':
    # 测试
    test_cases = [
        ("x0", 1),
        ("x0 + x1", 3),
        ("x0 * 2 + x1", 5),
        ("(x0 + x1) * x2", 5),
        ("sqrt(x0) + log(x1)", 5),
        ("x0 ** 2 + x1 ** 2", 7),
    ]
    
    print("表达式复杂度测试:")
    for expr, expected in test_cases:
        actual = count_ast_nodes(expr)
        status = "✓" if actual == expected else f"✗ (期望{expected})"
        print(f"  {expr:<30} -> {actual:3d} {status}")
    
    print("\n方法复杂度估算:")
    for method in ['buy_and_hold', 'macd', 'lightgbm', 'lstm', 'finrl_ppo']:
        complexity = estimate_method_complexity(method)
        print(f"  {method:<20} -> {complexity:4d}")
