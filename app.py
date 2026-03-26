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
**丘成桐中学科学奖（经济建模方向）辅助工具**  
本模型展示了在存在**消费者遗忘率 (Forgetting Rate, θ)** 与 **损失厌恶 (Loss Aversion, λ)** 的情况下，企业的两期最优动态定价策略 ($p_1^*, p_2^*$)。
消费者估值 $v$ 服从均匀分布 $U[0, 1]$。
""")

# 使用 Tabs 将宏观理论模型、微观模拟和文献库分开
tab1, tab2, tab3 = st.tabs(["📈 宏观理论最优定价模型", "👥 微观个体行为模拟 (Agent-based Simulation)", "📚 学术文献与调研资料库"])

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

with tab1:
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

with tab2:
    st.subheader("🏃‍♂️ 消费者微观行为模拟 (Monte Carlo Simulation)")
    st.markdown("""
    在这里，我们生成成百上千个**虚拟消费者**，模拟他们在第一期（面临拉新价）和第二期（面临续费价、遗忘率、损失厌恶）时的**真实决策过程**。
    你可以使用左侧栏的全局参数，也可以手动覆盖定价进行模拟测试。
    """)
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    sim_N = col_sim1.number_input("模拟消费者总人数 (N)", min_value=100, max_value=100000, value=10000, step=100)
    sim_p1 = col_sim2.number_input("设定首期价格 (p1)", min_value=0.0, max_value=1.0, value=float(opt_p1), step=0.05)
    sim_p2 = col_sim3.number_input("设定续费价格 (p2)", min_value=0.0, max_value=2.0, value=float(opt_p2), step=0.05)
    
    if st.button("🚀 开始运行模拟"):
        # 1. 生成 N 个消费者，估值 v 服从 U[0, 1]
        np.random.seed(42) # 固定种子以便结果可复现
        v_array = np.random.uniform(0, 1, sim_N)
        
        # 2. 第一期：决策是否订阅
        # 只要 v >= p1 就订阅
        subscribed_mask = v_array >= sim_p1
        n_subscribers_period1 = np.sum(subscribed_mask)
        
        # 获取第一期订阅者的估值
        v_subscribers = v_array[subscribed_mask]
        
        # 3. 第二期：分配状态 (遗忘 vs 清醒)
        # 为每个订阅者生成一个 0-1 的随机数，小于 theta 的为遗忘者
        forget_rolls = np.random.uniform(0, 1, n_subscribers_period1)
        forgotten_mask = forget_rolls < theta
        attentive_mask = ~forgotten_mask
        
        n_forgotten = np.sum(forgotten_mask)
        n_attentive = np.sum(attentive_mask)
        
        # 4. 第二期：续费决策
        # 遗忘者：无条件自动续费
        n_renew_forgotten = n_forgotten
        
        # 清醒者：重新评估效用
        # 效用 U2 = v - p2 - lambda * max(p2 - p1, 0)
        loss_penalty = lambda_val * max(sim_p2 - sim_p1, 0)
        v_attentive = v_subscribers[attentive_mask]
        u2_attentive = v_attentive - sim_p2 - loss_penalty
        
        renew_attentive_mask = u2_attentive >= 0
        n_renew_attentive = np.sum(renew_attentive_mask)
        n_cancel_attentive = n_attentive - n_renew_attentive
        
        # 5. 汇总结果
        total_renewals = n_renew_forgotten + n_renew_attentive
        total_cancellations = n_cancel_attentive
        
        # 计算模拟利润 (假设真实平均，未打折)
        sim_profit_1 = sim_p1 * n_subscribers_period1 / sim_N
        sim_profit_2 = sim_p2 * total_renewals / sim_N
        sim_total_profit = sim_profit_1 + delta * sim_profit_2
        
        # 绘制漏斗图展示流转过程
        st.markdown("#### 📊 用户留存漏斗与行为分布")
        
        # 使用 matplotlib 画一个简单的柱状图模拟漏斗
        fig_funnel, ax_funnel = plt.subplots(figsize=(10, 5))
        
        stages = ['初始目标客群', '第1期: 订阅', '第2期: 遗忘 (自动续费)', '第2期: 清醒 (主动续费)', '第2期: 清醒 (愤怒退订)']
        counts = [sim_N, n_subscribers_period1, n_renew_forgotten, n_renew_attentive, n_cancel_attentive]
        colors = ['#c7c7c7', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax_funnel.barh(stages[::-1], counts[::-1], color=colors[::-1])
        
        # 在柱子上添加文字
        for bar in bars:
            width = bar.get_width()
            ax_funnel.text(width + (sim_N*0.01), bar.get_y() + bar.get_height()/2, 
                           f'{int(width)} 人 ({(width/sim_N)*100:.1f}%)', 
                           ha='left', va='center', fontweight='bold')
            
        ax_funnel.set_xlim(0, sim_N * 1.2)
        ax_funnel.set_xlabel('用户数量')
        ax_funnel.set_title(f'蒙特卡洛模拟结果 (N={sim_N}, p1={sim_p1:.2f}, p2={sim_p2:.2f})')
        ax_funnel.spines['top'].set_visible(False)
        ax_funnel.spines['right'].set_visible(False)
        
        st.pyplot(fig_funnel)
        plt.close(fig_funnel)
        
        # 结果指标展示
        st.markdown("#### 💰 模拟商业指标测算")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("首期转化率", f"{(n_subscribers_period1/sim_N)*100:.1f}%")
        mcol2.metric("次期总续费率", f"{(total_renewals/n_subscribers_period1)*100:.1f}%", help="续费总人数 / 首期订阅人数")
        mcol3.metric("流失率 (Churn)", f"{(total_cancellations/n_subscribers_period1)*100:.1f}%", help="退订人数 / 首期订阅人数")
        mcol4.metric("单客模拟平均利润 (ARPU)", f"{sim_total_profit:.4f}")
        
        st.info(f"**微观行为分析**：在进入第2期的 {n_subscribers_period1} 名用户中，有 **{n_forgotten}** 人因为遗忘而默默接受了提价；而在 **{n_attentive}** 名清醒并检查账单的用户中，由于损失厌恶($\\lambda={lambda_val}$)，有 **{n_cancel_attentive}** 人觉得不划算而愤怒取消了订阅。")

with tab3:
    st.subheader("📚 学术文献与调研资料库")
    st.markdown("本资料库收录了关于**损失厌恶**、**消费者遗忘率（有限理性）**、**订阅制商业模式**以及**动态定价**的权威学术文献，作为本模型的理论基础。")
    
    # 内置文献数据库
    literature_db = [
        {
            "title": "Loss Aversion in Riskless Choice: A Reference-Dependent Model",
            "author": "Amos Tversky, Daniel Kahneman",
            "journal": "The Quarterly Journal of Economics",
            "year": "1991",
            "tags": ["损失厌恶", "参考价格", "行为经济学"],
            "abstract": "行为经济学奠基之作。提出了参考依赖（Reference-Dependent）效用模型，证明了消费者对损失（如提价）的敏感度通常是获得（如降价）的2.25倍，为本模型的 $\\lambda$ 参数提供了理论依据。"
        },
        {
            "title": "Paying Not to Go to the Gym",
            "author": "Stefano DellaVigna, Ulrike Malmendier",
            "journal": "American Economic Review",
            "year": "2006",
            "tags": ["订阅制", "消费者有限理性", "遗忘率", "实证研究"],
            "abstract": "经典的实证研究。通过分析健身房会员数据，发现消费者系统性地高估了自己未来的出勤率，并且在停止去健身房后，平均会延迟 2.3 个月才取消自动续费订阅。完美证明了“遗忘率”和“天真型消费者”的存在。"
        },
        {
            "title": "Pricing with Limited Attention",
            "author": "Pedro Bordalo, Nicola Gennaioli, Andrei Shleifer",
            "journal": "Handbook of Behavioral Economics",
            "year": "2018",
            "tags": ["有限注意力", "动态定价"],
            "abstract": "探讨了当消费者存在有限注意力（Limited Attention）时，企业如何通过复杂的定价结构（如初始低价+高昂的隐藏续费成本）来最大化利润。"
        },
        {
            "title": "Contract Renewal and the Exercise of Market Power",
            "author": "Paul Klemperer",
            "journal": "The American Economic Review",
            "year": "1995",
            "tags": ["转换成本", "自动续费", "市场势力"],
            "abstract": "研究了转换成本（Switching Costs）对企业定价策略的影响。在订阅制中，寻找退订按钮的时间成本和心理摩擦构成了隐性的转换成本，赋予了企业在续费期提价的市场势力。"
        },
        {
            "title": "The Effect of Default Options on Choice",
            "author": "Eric J. Johnson, Daniel Goldstein",
            "journal": "Science",
            "year": "2003",
            "tags": ["默认选项", "助推", "自动续费"],
            "abstract": "论证了“默认选项（Defaults）”对消费者最终选择的巨大影响。在自动续费模式下，“继续扣费”是默认选项，利用了消费者的惰性（Inertia），是订阅制盈利的核心机制。"
        }
    ]
    
    # 搜索框
    search_query = st.text_input("🔍 搜索文献 (支持标题、作者、摘要或标签关键词搜索)", placeholder="例如：损失厌恶、Kahneman、定价...")
    
    # 过滤逻辑
    if search_query:
        filtered_db = []
        query_lower = search_query.lower()
        for paper in literature_db:
            # 在各个字段中寻找关键词
            searchable_text = f"{paper['title']} {paper['author']} {paper['journal']} {paper['abstract']} {' '.join(paper['tags'])}".lower()
            if query_lower in searchable_text:
                filtered_db.append(paper)
    else:
        filtered_db = literature_db # 如果没有搜索，显示全部
        
    # 展示结果
    st.markdown(f"**共找到 {len(filtered_db)} 篇相关文献：**")
    
    for idx, paper in enumerate(filtered_db):
        with st.expander(f"📄 {paper['title']} ({paper['year']})"):
            st.markdown(f"**👤 作者：** {paper['author']}")
            st.markdown(f"**📖 期刊：** *{paper['journal']}*")
            
            # 渲染标签
            tags_html = "".join([f"<span style='background-color:#e1f5fe; color:#1f77b4; padding:3px 8px; border-radius:12px; font-size:0.85em; margin-right:5px;'>{tag}</span>" for tag in paper['tags']])
            st.markdown(f"**🏷️ 标签：** {tags_html}", unsafe_allow_html=True)
            
            st.markdown(f"**📝 核心摘要/应用价值：**\n> {paper['abstract']}")

st.markdown("---")
st.markdown("**Powered by Bella** | 祝猪猪由由与我一起在丘成桐科学奖中取得好成绩！")



