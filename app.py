import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
# Use Agg backend to prevent issues in Streamlit
matplotlib.use('Agg')

st.set_page_config(layout="wide", page_title="订阅服务动态定价模型")

st.title("📊 订阅服务自动续费动态定价可视化模型")
st.markdown("""
本模型展示了在存在**消费者遗忘率 (Forgetting Rate, θ)** 与 **损失厌恶 (Loss Aversion, λ)** 的情况下，企业的两期最优动态定价策略 ($p_1^*, p_2^*$)。
消费者估值 $v$ 服从均匀分布 $U[0, 1]$。
""")

st.sidebar.header("⚙️ 核心参数调节")
theta = st.sidebar.slider("遗忘率 (θ)", min_value=0.0, max_value=1.0, value=0.40, step=0.05, help="消费者在第二期忘记取消订阅的概率。")
lambda_val = st.sidebar.slider("损失厌恶系数 (λ)", min_value=0.0, max_value=5.0, value=2.25, step=0.1, help="提价时消费者感知的额外心理损失乘数。行为经济学经典值为2.25。")
delta = st.sidebar.slider("时间贴现因子 (δ)", min_value=0.5, max_value=1.0, value=0.90, step=0.05, help="企业对未来利润的折现率。")

def calc_profit(p1, p2, theta, lambda_val, delta):
    """
    计算给定定价策略 (p1, p2) 下的总利润
    """
    # 基础约束
    if p1 < 0 or p1 > 1 or p2 < 0:
        return 0
    
    # 第一期 (拉新期)
    # 消费者需求 D1 = 1 - p1
    D1 = max(0, 1 - p1)
    pi1 = p1 * D1
    
    # 第二期 (续费期)
    if p2 >= p1:
        # 提价情况 (或不变)
        # 保持清醒的消费者需要满足： v - p2 - λ(p2 - p1) >= 0
        v_thresh = p2 + lambda_val * (p2 - p1)
        # 能够续费的清醒消费者比例
        D2_att = max(0, 1 - v_thresh) if v_thresh < 1 else 0
        
        # 总需求 = 遗忘者需求 + 清醒者需求
        D2 = theta * D1 + (1 - theta) * D2_att
    else:
        # 降价情况 (续费用户的损失厌恶不会被触发，这里简化为原用户全留存)
        # 注意：在标准续费模型中，降价不会带来新客(因为已经过了拉新期)
        D2 = D1
        
    pi2 = p2 * D2
    
    # 总利润
    return pi1 + delta * pi2

def objective(p):
    # scipy 的 minimize 是求最小值，所以取负
    return -calc_profit(p[0], p[1], theta, lambda_val, delta)

# 求解当前参数下的最优定价 (使用 L-BFGS-B 或 SLSQP)
res = minimize(objective, [0.3, 0.5], bounds=[(0, 1), (0, 2)], method='SLSQP')
opt_p1, opt_p2 = res.x
max_pi = -res.fun

st.subheader("💡 1. 当前参数下的最优定价结果")
col1, col2, col3, col4 = st.columns(4)
col1.metric("首期拉新价 $p_1^*$", f"{opt_p1:.4f}")
col2.metric("次期续费价 $p_2^*$", f"{opt_p2:.4f}")
col3.metric("提价幅度 $\Delta p$", f"{opt_p2 - opt_p1:.4f}")
col4.metric("企业最大总利润 $\Pi^*$", f"{max_pi:.4f}")

# 经济学洞察提示
if opt_p1 < 0.4 and opt_p2 > 0.5:
    st.success("📝 **经济学洞察**：当前呈现出典型的 **“诱捕定价 (Bait-and-Snatch)”** 特征！首期以低价甚至亏本获客，次期利用消费者的遗忘率高价收割。")
elif opt_p2 - opt_p1 < 0.05:
    st.warning("📝 **经济学洞察**：由于损失厌恶过强或遗忘率极低，企业被迫采取 **“平滑定价 (Price Smoothing)”**，不敢大幅提价，以免激怒清醒用户。")
else:
    st.info("📝 **经济学洞察**：企业在“收割遗忘者”和“保留清醒者”之间找到了平衡，采取了温和的提价策略。")

st.markdown("---")
st.subheader("📈 2. 核心关系可视化 (Comparative Statics)")

col_fig1, col_fig2 = st.columns(2)

# 绘图1：遗忘率 θ 对最优定价的影响
with col_fig1:
    st.markdown("##### 遗忘率 θ 如何影响定价策略？")
    st.markdown("*(假设损失厌恶 λ 固定为当前选择的值)*")
    thetas = np.linspace(0, 1, 25)
    p1_list, p2_list = [], []
    for t in thetas:
        res_t = minimize(lambda p: -calc_profit(p[0], p[1], t, lambda_val, delta), [0.3, 0.5], bounds=[(0, 1), (0, 2)])
        p1_list.append(res_t.x[0])
        p2_list.append(res_t.x[1])
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(thetas, p1_list, label='$p_1^*$ (首期拉新价)', color='#1f77b4', marker='o', markersize=4)
    ax1.plot(thetas, p2_list, label='$p_2^*$ (次期续费价)', color='#d62728', marker='s', markersize=4)
    ax1.set_xlabel('遗忘率 θ (Forgetting Rate)')
    ax1.set_ylabel('最优价格')
    ax1.set_title(f'遗忘率与最优定价关系 (λ={lambda_val:.2f})')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)
    plt.close(fig1)

# 绘图2：损失厌恶 λ 对最优定价的影响
with col_fig2:
    st.markdown("##### 损失厌恶 λ 如何影响定价策略？")
    st.markdown("*(假设遗忘率 θ 固定为当前选择的值)*")
    lambdas = np.linspace(0, 5, 25)
    p1_list_l, p2_list_l = [], []
    for l in lambdas:
        res_l = minimize(lambda p: -calc_profit(p[0], p[1], theta, l, delta), [0.3, 0.5], bounds=[(0, 1), (0, 2)])
        p1_list_l.append(res_l.x[0])
        p2_list_l.append(res_l.x[1])
        
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(lambdas, p1_list_l, label='$p_1^*$ (首期拉新价)', color='#1f77b4', marker='o', markersize=4)
    ax2.plot(lambdas, p2_list_l, label='$p_2^*$ (次期续费价)', color='#d62728', marker='s', markersize=4)
    ax2.set_xlabel('损失厌恶系数 λ (Loss Aversion)')
    ax2.set_ylabel('最优价格')
    ax2.set_title(f'损失厌恶与最优定价关系 (θ={theta:.2f})')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.subheader("🌐 3. 利润曲面 3D 视图 (Profit Surface)")
st.markdown("直观展示在当前参数下，不同的定价组合 $(p_1, p_2)$ 带来的总利润变化。红色圆点为最大利润点。")

# 生成网格数据
p1_grid = np.linspace(0.01, 0.99, 40)
p2_grid = np.linspace(0.01, 1.20, 40)
P1, P2 = np.meshgrid(p1_grid, p2_grid)
Pi = np.zeros_like(P1)

for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        Pi[i, j] = calc_profit(P1[i, j], P2[i, j], theta, lambda_val, delta)

fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111, projection='3d')
# 绘制曲面
surf = ax3.plot_surface(P1, P2, Pi, cmap='viridis', edgecolor='none', alpha=0.85)
# 标记最优点
ax3.scatter([opt_p1], [opt_p2], [max_pi], color='red', s=100, label='最优定价点 (Optimal Point)', zorder=5)

ax3.set_xlabel('首期价格 $p_1$')
ax3.set_ylabel('续费价格 $p_2$')
ax3.set_zlabel('总利润 $\Pi$')
ax3.set_title('总利润随定价变化的 3D 曲面图')
ax3.legend()
fig3.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

# 将 3D 图表展示在居中的位置
col_3d1, col_3d2, col_3d3 = st.columns([1, 4, 1])
with col_3d2:
    st.pyplot(fig3)
plt.close(fig3)


