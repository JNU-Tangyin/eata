import numpy as np
import logging
import os

# 创建日志目录
os.makedirs('logs', exist_ok=True)

# 配置日志记录 - 降低默认级别至WARNING以减少输出
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('logs/mcts_adapter.log'),  # 文件处理器
                        logging.StreamHandler()  # 控制台处理器
                    ])
logger = logging.getLogger('MCTSAdapter')
logger.setLevel(logging.WARNING)  # 只记录警告及以上级别的消息

class MCTSAdapter:
    """处理MCTS与神经网络之间的维度不匹配问题"""
    
    @staticmethod
    def align_policy(policy_nn, policy_ucb, grammar_vocab, mcts_grammars):
        """
        将神经网络策略(policy_nn)与MCTS策略(policy_ucb)的维度对齐
        
        Args:
            policy_nn: 神经网络输出的策略向量，维度为[len(grammar_vocab)]
            policy_ucb: MCTS计算的UCB策略向量，维度为[len(mcts_grammars)]
            grammar_vocab: 神经网络使用的语法词汇表列表
            mcts_grammars: MCTS使用的语法列表
            
        Returns:
            aligned_policy_nn: 与policy_ucb维度一致的神经网络策略
        """
        # 输出维度信息用于调试
        logger.info(f"Policy NN shape: {policy_nn.shape}, Policy UCB shape: {policy_ucb.shape}")
        logger.info(f"Grammar vocab length: {len(grammar_vocab)}, MCTS grammars length: {len(mcts_grammars)}")
        
        # 创建一个与policy_ucb相同维度的新向量
        aligned_policy_nn = np.zeros_like(policy_ucb)
        
        # 创建从grammar到索引的映射
        vocab_idx_map = {grammar: idx for idx, grammar in enumerate(grammar_vocab)}
        
        # 记录映射到的条目数量
        mapped_count = 0
        
        # 将policy_nn中的值映射到aligned_policy_nn中对应位置
        for i, grammar in enumerate(mcts_grammars):
            if grammar in vocab_idx_map:
                # 如果MCTS语法在神经网络词汇表中存在，取对应值
                vocab_idx = vocab_idx_map[grammar]
                if vocab_idx < len(policy_nn):
                    aligned_policy_nn[i] = policy_nn[vocab_idx]
                    mapped_count += 1
        
        # 记录映射率
        mapping_ratio = mapped_count / len(mcts_grammars) * 100 if mcts_grammars else 0
        logger.info(f"Mapped {mapped_count} out of {len(mcts_grammars)} grammars ({mapping_ratio:.2f}%)")
        
        # 归一化确保和为1
        if aligned_policy_nn.sum() > 0:
            aligned_policy_nn = aligned_policy_nn / aligned_policy_nn.sum()
        else:
            # 如果全为零，使用均匀分布
            logger.warning("All values in aligned_policy_nn are zero, using uniform distribution")
            aligned_policy_nn = np.ones_like(aligned_policy_nn) / len(aligned_policy_nn)
            
        return aligned_policy_nn
    
    @staticmethod
    def patch_mcts(mcts_instance, network):
        """
        修补MCTS类的run方法，解决维度不匹配问题
        
        Args:
            mcts_instance: MCTS实例
            network: 神经网络上下文(p_v_net_ctx)
        """
        logger.info(f"Patching MCTS instance {id(mcts_instance)} with network {id(network)}")
        
        # 对象已经有适配器
        if hasattr(mcts_instance, '_mcts_adapter_patched') and mcts_instance._mcts_adapter_patched:
            logger.info("MCTS instance already patched, skipping")
            return mcts_instance
        
        # 保存原始的get_policy3方法
        original_get_policy3 = mcts_instance.get_policy3
        
        # 创建适配后的get_policy3方法
        def adapted_get_policy3(nA, UC, seq, state, net, softmax=True):
            # 调用原始方法获取policy_nn
            policy_nn, value_nn = original_get_policy3(nA, UC, seq, state, net, softmax)
            
            # 确保policy_nn维度与MCTS内部一致
            if len(policy_nn) != nA and hasattr(net, 'grammar_vocab'):
                logger.info(f"Dimension mismatch: policy_nn has {len(policy_nn)} elements, but nA={nA}")
                # 获取神经网络词汇表
                grammar_vocab = net.grammar_vocab
                # 获取MCTS语法列表
                mcts_grammars = mcts_instance.grammars
                
                # 调整policy_nn维度
                try:
                    policy_nn = MCTSAdapter.align_policy(
                        policy_nn, 
                        np.zeros(nA),  # 创建与nA一致的空向量
                        grammar_vocab, 
                        mcts_grammars
                    )
                    logger.info(f"Successfully aligned policy_nn to shape {policy_nn.shape}")
                except Exception as e:
                    logger.error(f"Error aligning policy: {e}")
                    # 出错时使用均匀分布
                    policy_nn = np.ones(nA) / nA
            
            return policy_nn, value_nn
        
        # 替换MCTS实例的get_policy3方法
        mcts_instance.get_policy3 = adapted_get_policy3
        
        # 标记为已经适配
        mcts_instance._mcts_adapter_patched = True
        logger.info(f"MCTS instance {id(mcts_instance)} successfully patched")
        
        # 返回修补后的实例
        return mcts_instance
        
    @staticmethod
    def patch_engine(engine):
        """一键修补引擎的MCTS适配器"""
        if hasattr(engine, 'model') and hasattr(engine.model, 'p_v_net_ctx'):
            try:
                # 每次创建MCTS时都进行适配
                original_run = engine.model.run
                
                def patched_run(X, y=None, force_alpha=None):
                    logger.info(f"Running engine model with patched MCTS adapter")
                    # 检查原始方法的参数签名
                    import inspect
                    sig = inspect.signature(original_run)
                    param_count = len(sig.parameters)
                    
                    # 根据原始方法参数数量正确调用
                    if param_count <= 2:
                        # 只接受X参数或X,y参数
                        result = original_run(X) if y is None else original_run(X, y)
                    else:
                        # 支持额外的force_alpha参数
                        logger.info(f"使用force_alpha={force_alpha}调用run方法")
                        if force_alpha is not None:
                            # 这里需要修改MCTS核心代码，不在这里实现
                            result = original_run(X, y)
                        else:
                            result = original_run(X, y)
                    return result
                
                engine.model.run = patched_run
                logger.info(f"Engine {id(engine)} successfully patched with MCTS adapter")
                return True
            except Exception as e:
                logger.error(f"Error patching engine: {e}")
                return False
        else:
            logger.error(f"Engine {id(engine)} missing model or p_v_net_ctx")
            return False
