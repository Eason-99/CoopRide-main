import torch
from log import Logger
import json
from openai import OpenAI
from zai import ZhipuAiClient

# ==================== OpenAI API 配置 ====================
# 默认 API 配置（可在代码中修改或通过参数传入覆盖）
DEFAULT_OPENAI_API_KEY = "sk-yXEDn5KuOGgLa83lGEFmxWoSPkGMvtfje14fLrM228ttHX3K"  # 请替换为你的实际 API Key
DEFAULT_OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"  # 国内
# DEFAULT_OPENAI_BASE_URL = "https://api.chatanywhere.org/v1"  # 国外
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # 默认使用的模型

DEFAULT_ZHIPU_API_KEY = "fe0b7f61295f44918c8cf557bb5ac8d1.JPzOXBpZJ1DGKUcl"
DEFAULT_ZHIPU_MODEL = "glm-4.7"

# ==================== 函数定义 ====================

def get_env_description():
    """
    获取环境描述字符串 (针对 LLM 逻辑推理优化的版本)
    """
    description = (
        "The environment is a high-fidelity, multi-agent ride-hailing simulator. "
        "The goal is to find an optimal dispatch policy by weighting 10 features to maximize "
        "long-term system efficiency, order response rates (ORR), and total revenue (GMV).\n"
        "### Core Optimization Logic:\n"
        "The features are categorized into four dimensions that the LLM must balance:\n\n"
        "1. **Supply-Demand Balancing (Spatial Equilibrium)**:\n"
        "   - Destination Supply Shortage & Global Imbalance: Used to relocate drivers from surplus "
        "areas to high-demand, low-supply regions to prevent local vehicle stagnation.\n"
        "   - Current Grid Push & Destination Order Pull: Dynamic forces that 'push' idle drivers "
        "away from congested grids and 'pull' them toward order-dense hotspots.\n\n"
        "2. **Immediate Economic Returns (Short-term Profit)**:\n"
        "   - Order Price & Price Efficiency: Direct indicators of trip value. Efficiency (Price/Duration) "
        "is critical for maximizing driver hourly income.\n\n"
        "3. **Operational Constraints & Costs**:\n"
        "   - Duration Cost & Cross-Grid Penalty: Represent the negative utility of long pickups "
        "and movements that cross administrative boundaries, which often increase system entropy.\n"
        "   - Real Order Bias: Distinguishes between actual passenger orders and virtual re-positioning tasks.\n\n"
        "4. **Future Value (Reinforcement Learning)**:\n"
        "   - MDP Advantage: The V-value (long-term discounted reward). This is the key bridge between "
        "greedy immediate matching and strategic future positioning.\n\n"
        "### Challenge:\n"
        "The optimizer must mitigate the 'Greedy Trap' (over-focusing on current Price) while avoiding "
        "systemic instability caused by excessive relocation (over-focusing on Shortage)."
    )
    return description

def format_history_for_llm(topK_records):
    """
    将历史记录列表格式化为 Markdown 表格，供 LLM 分析
    """
    if not topK_records:
        return "No historical data available yet."
    
    header = "| Iteration | ORR (Efficiency) | GMV (Revenue) | Weights (w0 to w9) |\n"
    separator = "| :--- | :--- | :--- | :--- |\n"
    rows = ""
    
    for i, rec in enumerate(topK_records):
        # 标注记录类型
        tag = "Current" if i == 0 else ("Top ORR" if i <= 3 else "Top GMV")
        w_str = "[" + ", ".join([f"{w:.2f}" for w in rec['weights']]) + "]"
        rows += f"| {rec['step_num']} ({tag}) | {rec['orr']:.4f} | {rec['gmv']:.2f} | {w_str} |\n"
    
    return header + separator + rows

def build_weight_optimization_prompt(weights_list, env_description=None, max_steps=40, 
                                      step_number=0, episode_reward_topK=None, step_size=1.0):
    """
    Build the weight optimization prompt with RL optimization framework
    
    Args:
        weights_list: list, containing 10 weight values
        env_description: str, description of the environment (optional)
        max_steps: int, maximum number of optimization steps (default: 40)
        step_number: int, current iteration number (default: 0)
        episode_reward_topK: str, historical params and rewards buffer (optional)
        step_size: float, maximum step size for each adjustment (default: 1.0)
    
    Returns:
        str: The constructed prompt string
    """
    # Format episode reward buffer if provided
    buffer_string = format_history_for_llm(episode_reward_topK)
    
    prompt = f"""You are a global policy optimizer, helping me find the global optimal policy for a ride-hailing dispatch system.

# Environment:
{env_description if env_description else "Ride-hailing dispatch optimization using 10 feature weights."}

# Regarding the parameters **params**:
**params** is an array of 10 float numbers representing the weight vector.
**params** values are in the range of [-5.0, 5.0] with 4 decimal places.
params represent a linear policy. f(params) is the episodic reward of the dispatch policy.

# Current weight configuration:
w0 (Destination Supply Shortage): {weights_list[0]:.4f} - Destination driver shortage (End - Start), positive values incentivize going to driver-scarce areas
w1 (Order Price): {weights_list[1]:.4f} - Order fare, positive values incentivize high-value orders
w2 (Duration Cost): {weights_list[2]:.4f} - Trip duration penalty, negative values penalize long-distance trips
w3 (Price Efficiency): {weights_list[3]:.4f} - Efficiency (fare per minute), positive values incentivize high-efficiency orders
w4 (Real Order Bias): {weights_list[4]:.4f} - Real order bias, positive values prioritize real orders over virtual dispatch
w5 (Current Grid Push): {weights_list[5]:.4f} - Current grid congestion push, positive values push away from congested areas
w6 (Destination Order Pull): {weights_list[6]:.4f} - Destination order pull, positive values attract to order-dense areas
w7 (Cross-Grid Penalty): {weights_list[7]:.4f} - Cross-grid penalty, negative values penalize cross-grid dispatch
w8 (Global Imbalance): {weights_list[8]:.4f} - Global imbalance correction
w9 (MDP Advantage): {weights_list[9]:.4f} - Long-term value (V-value), positive values consider future benefits

# Historical Performance & Pareto Frontier:
The following table summarizes the performance of previous weight configurations. It includes the **Immediate Previous Trial** (for local gradient-like adjustment) and the **Best-In-Class Records** (representing the current Pareto frontier of the system).

{buffer_string}

### Evaluation Metrics:
- **Order Response Rate (ORR)**: Reflects system efficiency and passenger satisfaction. High ORR indicates a balanced supply-demand state where most orders are serviced.
- **Gross Merchandise Volume (GMV)**: Reflects total economic throughput. High GMV indicates that the policy successfully prioritizes high-value orders or high-efficiency trips.

### Analytical Task for History:
1. **Correlation Analysis**: Identify which weight adjustments led to a simultaneous increase in both ORR and GMV, or where a trade-off occurred.
2. **Failure Mode Identification**: If the Current Trial (Iteration {step_number}) shows a significant drop in metrics compared to Top Records, diagnose which weights in the current vector might be causing the instability or "Greedy Trap."
3. **Weight Sensitivity**: Observe the variance in weights among Top ORR vs. Top GMV records to determine which parameters are most sensitive to specific KPIs.

# Optimization process:
- We are at iteration {step_number} out of {max_steps}
- During exploration, use search step size of {step_size}
- Do not propose previously seen params
- The global optimum requires exploration; continue exploring if you suspect local optimum
- Balance between exploration (trying new regions) and exploitation (refining good regions)

# Task: Based on current weight configuration and historical performance, analyze potential issues and provide optimization suggestions.

# Please respond in the following format:

1. **Current Weight Analysis**: Briefly describe the characteristics and potential issues of the current weight configuration
2. **Optimization Suggestions**: Provide specific optimization directions and suggested weight adjustments
3. **Suggested Weights**: Provide an optimized weight array in JSON format:
{{
    "suggested_weights": [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9]
}}
    Suggested range for each weight: [-5.0, 5.0]

Note: Keep weight values within reasonable ranges to avoid extreme values that could destabilize the system.
"""
    return prompt


def call_openai_for_weight_optimization(llm_optimized_weights, api_key=None, base_url=None, model=None, 
                                        max_steps=10, step_number=0, episode_reward_topK=None):
    """
    通过 OpenAI 接口调用大模型获取权重优化建议
    
    Args:
        llm_optimized_weights: torch.Tensor, 形状为 (10,) 的权重向量
        api_key: str, OpenAI API 密钥（如果不提供，则使用 DEFAULT_OPENAI_API_KEY 或从环境变量 OPENAI_API_KEY 读取）
        base_url: str, OpenAI API base URL（如果不提供，则使用 DEFAULT_OPENAI_BASE_URL 或默认官方端点）
        model: str, 使用的 OpenAI 模型名称（如果不提供，则使用 DEFAULT_OPENAI_MODEL）
        max_steps: int, 最大优化步数（默认：400）
        step_number: int, 当前迭代次数（默认：0）
        episode_reward_topK: str, 历史参数与奖励缓冲区（可选）
        step_size: float, 每次调整的最大步幅（默认：0.5）
    
    Returns:
        dict: 包含大模型回复和建议权重的字典
            - 'response': str, 大模型的完整回复
            - 'suggested_weights': torch.Tensor or None, 大模型建议的权重向量（如果解析成功）
            - 'suggestion': str, 大模型的主要建议摘要
    """
    # 将 PyTorch tensor 转换为列表
    if isinstance(llm_optimized_weights, torch.Tensor):
        weights_list = llm_optimized_weights.tolist()
    else:
        weights_list = llm_optimized_weights
    
    # 构建提示词
    prompt = build_weight_optimization_prompt(
        weights_list=weights_list,
        env_description=get_env_description(),
        max_steps=max_steps,
        step_number=step_number,
        episode_reward_topK=episode_reward_topK
    )
    
    # 确定使用的 API 配置
    final_api_key = api_key if api_key else DEFAULT_OPENAI_API_KEY
    final_base_url = base_url if base_url else DEFAULT_OPENAI_BASE_URL
    final_model = model if model else DEFAULT_OPENAI_MODEL
    
    try:
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key=final_api_key,
            base_url=final_base_url
        )
        
        Logger.info("正在调用 OpenAI API 进行权重优化...")
        Logger.info(f"当前权重: {weights_list}")
        Logger.info(f"当前迭代: {step_number}/{max_steps}")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=final_model,
            messages=[
                {"role": "system", "content": "You are a professional ride-hailing dispatch system optimization expert, skilled in analyzing and optimizing dispatch weight parameters through iterative exploration and exploitation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=12800
        )
        
        # 获取回复内容
        llm_response = response.choices[0].message.content
        Logger.info(f"LLM 回复: {llm_response}")
        
        # 尝试解析 JSON 格式的建议权重
        suggested_weights = None
        suggestion = ""
        
        try:
            # 查找 JSON 格式的权重
            if "suggested_weights" in llm_response:
                # 提取 JSON 部分
                import re
                json_pattern = r'\{[^}]*"suggested_weights"\s*:\s*\[[^\]]+\][^}]*\}'
                json_match = re.search(json_pattern, llm_response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    suggested_weights = parsed["suggested_weights"]
                    
                    # 验证权重数量和范围
                    if len(suggested_weights) == 10:
                        # 限制权重范围
                        suggested_weights = [max(-5.0, min(5.0, w)) for w in suggested_weights]
                        suggestion = f"LLM建议优化权重为: {suggested_weights}"
                        Logger.info(f"成功解析建议权重: {suggested_weights}")
                    else:
                        Logger.warning(f"解析的权重数量错误，期望10个，实际{len(suggested_weights)}")
                        suggested_weights = None
        except Exception as e:
            Logger.warning(f"解析建议权重失败: {e}")
        
        # 如果未解析到权重，尝试提取建议摘要
        if not suggestion:
            lines = llm_response.split('\n')
            for line in lines:
                if '建议' in line or '优化' in line or '推荐' in line or 'suggest' in line.lower() or 'recommend' in line.lower():
                    suggestion = line.strip()
                    break
            if not suggestion:
                suggestion = llm_response[:100]  # 使用前100字符作为摘要
        
        return {
            'response': llm_response,
            'suggested_weights': torch.tensor(suggested_weights) if suggested_weights else None,
            'suggestion': suggestion
        }
        
    except Exception as e:
        # 打印详细的错误信息
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        
        print("\n" + "=" * 70)
        print("[ERROR] OpenAI API 调用失败详情")
        print("=" * 70)
        print(f"错误类型: {error_type}")
        print(f"错误信息: {error_msg}")
        print(f"使用的 Base URL: {final_base_url}")
        print(f"使用的模型: {final_model}")
        print(f"API Key 前缀: {final_api_key[:10]}..." if len(final_api_key) > 10 else "API Key: 未设置或长度不足")
        print("\n完整堆栈跟踪:")
        print(full_traceback)
        print("=" * 70)
        
        Logger.warning(f"调用 OpenAI API 失败: {error_type} - {error_msg}")
        return {
            'response': f"API 调用失败: {error_type} - {error_msg}",
            'suggested_weights': None,
            'suggestion': f"API 调用失败 ({error_type})，无法获取建议"
        }

def get_llm_optimized_weights():
    """
    模拟从 LLM 获取优化后的权重向量。
    在实际部署中，这里会读取 LLM 最新生成的 JSON 配置或数据库中的参数。
    
    返回的权重对应以下特征：
    0. Dest Supply Shortage (+)  : 目的地缺车程度 (越大越缺)
    1. Order Price (+)           : 订单金额
    2. Duration Cost (-)         : 耗时惩罚
    3. Price Efficiency (+)      : 效率 (元/分钟)
    4. Real Order Bias (+)       : 真实订单偏置 (是真订单为1，虚拟调度为0)
    5. Current Grid Push (+)     : 当前网格拥堵推力 (本地熵)
    6. Destination Order Pull (+) : 目的地订单拉力 (订单增量)
    7. Cross-Grid Penalty (-)    : 跨区惩罚 (跨区为1，本地为0)
    8. Global Imbalance (+)      : 全局不平衡度 (与全局熵的差异)
    9. MDP Advantage (+)         : 长期价值 (V值)
    """
    # 示例权重：LLM 可能会输出这样一组值
    # 注意：正负号代表了LLM对特征方向的理解
    # 例如 duration 是成本，权重可能是负数；但在特征工程中通常保留原始数值，由权重控制正负
    weights = torch.tensor(
        [
            0.0,  # w0: 缺口权重 (大幅鼓励去缺车的地方)
            0.0,  # w1: 价格权重
            0.0,  # w2: 耗时权重 (惩罚长距离)
            0.0,  # w3: 效率权重
            0.0,  # w4: 真实订单优先
            0.0,  # w5: 拥堵时推离本地
            0.0,  # w6: 去订单多的地方
            0.0,  # w7: 跨区微惩罚
            0.0,  # w8: 异常状态修正
            0.0,  # w9: 长期价值
        ]
    )
    return weights

def compute_llm_correction(llm_optimized_weights, state, order, mask, device='cpu'):
    """
    计算 LLM 优化的显式逻辑修正项 (Score_LLM)
    
    :param state: (Grid_Num, State_Dim)
    :param order: (Grid_Num, Max_Order_Num, Order_Dim)
    :param mask: (Grid_Num, Max_Order_Num) 掩码，虽然这里主要计算特征，mask可用于后续处理
    :param device: 计算设备
    :return: correction_logits: (Grid_Num, Max_Order_Num) 修正分数矩阵
    """
    grid_num, max_order_num, _ = order.shape
    
    # 特征提取 (Feature Extraction)
    # 我们需要构建一个特征张量 feats: (N, K, 10)
    
    # --- 第一梯队：决定性因子 ---
    
    # F0: Destination Supply Shortage (司机数量差: End - Start)
    # order[..., 6] 是 diff，值越大代表终点司机比起点多（不缺车）。
    # 我们定义的特征是"缺车程度"，所以取负。
    # 假设 CoopRide 的 order[..., 6] 定义确为 End - Start
    f0_shortage = -order[..., 6] 
    
    # F1: Order Price (订单价格)
    f1_price = order[..., 2]
    
    # F2: Duration Cost (耗时)
    f2_duration = order[..., 3]
    
    # F3: Price Efficiency (效率)
    # 加上 epsilon 防止除零
    epsilon = 1e-6
    f3_efficiency = f1_price / (f2_duration + epsilon)
    
    # --- 第二梯队：策略调节因子 ---
    
    # F4: Real Order Bias (是否真实订单)
    # order[..., 4] == -1 为真实订单
    f4_real_bias = (order[..., 4] == -1).float()
    
    # F5: Current Grid Push (当前网格拥堵度)
    # 来源 state[..., 4] (Local Entropy: idle / (real + idle))
    # 熵越大，代表空闲司机比例越高，越拥堵。
    # state 维度是 (N, Ds)，需要扩展到 (N, K)
    f5_push = state[..., 4].unsqueeze(-1).expand(-1, max_order_num)
    
    # F6: Destination Order Pull (目的地订单增量)
    # order[..., 7] 是 Order Num Diff (End - Start)
    f6_pull = order[..., 7]
    
    # --- 第三梯队：辅助因子 ---
    
    # F7: Cross-Grid Penalty (是否跨区)
    # order[..., 0] 是 Start Grid ID, order[..., 1] 是 End Grid ID
    # 注意：ID 可能是浮点数存储，比较时需谨慎，或直接由逻辑判断
    f7_cross = (order[..., 0] != order[..., 1]).float()
    
    # F8: Global Imbalance (全局不平衡度)
    # state[..., 5] 是 abs(local_entropy - global_entropy)
    f8_imbalance = state[..., 5].unsqueeze(-1).expand(-1, max_order_num)
    
    # F9: MDP Advantage (长期价值)
    # order[..., -1]
    f9_advantage = order[..., -1]
    
    # 特征堆叠
    # 将所有特征堆叠在最后一维 -> (N, K, 10)
    features = torch.stack([
        f0_shortage,
        f1_price,
        f2_duration,
        f3_efficiency,
        f4_real_bias,
        f5_push,
        f6_pull,
        f7_cross,
        f8_imbalance,
        f9_advantage
    ], dim=-1).to(device)
    
    # 假设没有传入权重，用默认值
    # weights: (10,)
    if llm_optimized_weights is None:
        llm_optimized_weights = get_llm_optimized_weights().to(device)
        Logger.warning("[WARNING!!!!!!!!!!!!!!!!!!] 使用默认 LLM 优化权重向量")
    
    # 计算线性修正项 (Dot Product)
    # (N, K, 10) * (10,) -> (N, K)
    # 这一步实现了 Score_LLM = sum(w_i * f_i)
    correction_logits = torch.matmul(features, llm_optimized_weights.to(device))
    
    # 可选：如果需要在 mask 无效订单位置设为负无穷，可以在这里做
    # 但通常这步是在 Softmax 之前统一做的，这里只返回 Raw Logits 修正值
    if mask is not None:
        # 将无效订单的修正值设为 0 (或者不处理，反正原 Logits 会被 Mask 掉)
        correction_logits = correction_logits * mask.float()
        
    return correction_logits

# --- 使用示例 ---
# 在你的神经网络 forward 或 action 函数中调用：
# logits_nn = self.compute_original_logits(...) # 原有的 Neural Network Logits
# logits_llm = compute_llm_correction(state, order, mask, device)
#
# # 混合残差控制 (lambda 系数可设为 1.0 或也是一个 LLM 参数)
# final_logits = logits_nn + 1.0 * logits_llm
# probs = torch.softmax(final_logits, dim=-1)


# ==================== 测试代码 ====================
def test_openai():
    print("=" * 70)
    print("测试 call_openai_for_weight_optimization 函数")
    print("=" * 70)
    print(f"\n默认配置:")
    print(f"  API Key: {'已设置' if DEFAULT_OPENAI_API_KEY != 'your-api-key-here' else '未设置 (使用环境变量)' }")
    print(f"  Base URL: {DEFAULT_OPENAI_BASE_URL if DEFAULT_OPENAI_BASE_URL else '使用默认端点'}")
    print(f"  Model: {DEFAULT_OPENAI_MODEL}")
    
    # 测试 1: 全零权重（使用默认配置）
    print("\n[测试 1] 使用全零权重（默认配置）...")
    zero_weights = torch.zeros(10)
    
    result1 = call_openai_for_weight_optimization(
        llm_optimized_weights=zero_weights
    )
    
    print(f"✓ 返回类型: {type(result1)}")
    print(f"✓ 返回键: {result1.keys()}")
    print(f"✓ 建议: {result1['suggestion']}")
    
    if result1['suggested_weights'] is not None:
        print(f"✓ 建议权重: {result1['suggested_weights'].tolist()}")
        print("✓ 测试 1 通过: 成功获取建议权重")
    else:
        print("✗ 测试 1 失败: 未能获取建议权重")
    
    # 测试 2: 非零权重（使用默认配置）
    print("\n[测试 2] 使用非零权重（默认配置）...")
    custom_weights = torch.tensor([
        1.0,   # w0
        0.5,   # w1
        -0.3,  # w2
        0.2,   # w3
        1.5,   # w4
        0.1,   # w5
        0.8,   # w6
        -0.5,  # w7
        0.0,   # w8
        0.3    # w9
    ])
    
    result2 = call_openai_for_weight_optimization(
        llm_optimized_weights=custom_weights
    )
    
    print(f"✓ 返回类型: {type(result2)}")
    print(f"✓ 返回键: {result2.keys()}")
    
    if result2['suggested_weights'] is not None:
        print(f"✓ 建议权重: {result2['suggested_weights'].tolist()}")
        print("✓ 测试 2 通过: 成功获取建议权重")
    else:
        print("✗ 测试 2 失败: 未能获取建议权重")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    

import traceback

def test_zhipu():
    print("=" * 70)
    print("[INFO] 开始测试 Zhipu API")
    print("=" * 70)
    
    try:
        # 初始化客户端
        print("[INFO] 正在初始化 ZhipuAiClient...")
        client = ZhipuAiClient(api_key=DEFAULT_ZHIPU_API_KEY)

        # 创建聊天完成请求
        print(f"[INFO] 正在发送请求 (模型: {DEFAULT_ZHIPU_MODEL})...")
        response = client.chat.completions.create(
            model=DEFAULT_ZHIPU_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手。"
                },
                {
                    "role": "user",
                    "content": "你好，请介绍一下自己。"
                }
            ],
            temperature=0.6
        )

        # 获取回复
        print("\n[SUCCESS] 请求成功！获取到的回复如下:")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)

    except Exception as e:
        print("\n" + "=" * 70)
        print("[ERROR] Zhipu API 调用失败详情")
        print("=" * 70)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n完整堆栈跟踪:")
        # 打印完整的调用栈，便于定位是网络层还是解析层的问题
        traceback.print_exc()
        
    finally:
        print("\n" + "=" * 70)
        print("[INFO] 测试结束")
        print("=" * 70)

if __name__ == "__main__":
    # test_openai()
    test_zhipu()
    