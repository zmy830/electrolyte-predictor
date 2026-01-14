# -*- coding: utf-8 -*-
"""
特征工程原理展示页面
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="特征原理 - 电解液预测",
    page_icon="🔬",
    layout="wide",
)

def main():
    st.title("🔬 特征工程原理")
    
    st.markdown("""
    本系统使用多层特征工程架构，将电解液的物理化学性质转换为机器学习模型可用的特征。
    """)
    
    # 特征层级
    st.header("📊 特征层级架构")
    
    layers = {
        "Layer 0: 基础物性": {
            "features": ["eps_mix", "ln_eta0_mix", "DN_mix", "AN_mix", "MW_mix", "rho_mix", "dipole_mix"],
            "description": "基于溶剂物性的加权平均",
            "formula": r"\varepsilon_{mix} = \sum_{i} w_i \cdot \varepsilon_i",
        },
        "Layer 1: 结构分组": {
            "features": ["frac_cyclic", "frac_linear", "frac_ether"],
            "description": "按溶剂结构类型统计分数",
            "formula": r"f_{cyclic} = \sum_{i \in cyclic} w_i",
        },
        "Layer 2: 协同效应": {
            "features": ["synergy_carb"],
            "description": "捕捉环状+链状共存时的电导率峰值",
            "formula": r"synergy = f_{cyclic} \times f_{linear}",
        },
        "Layer 3: 温度耦合": {
            "features": ["eps_over_T", "inv_T", "inv_TmT0", "ln_eta_ideal_T"],
            "description": "温度相关的物理量",
            "formula": r"\ln\eta_{ideal}(T) = A + \frac{B}{T}",
        },
        "Layer 4: 盐效应门控": {
            "features": ["salt_x_cyclic", "salt_x_linear_hi", "salt_x_linear_lo"],
            "description": "盐对粘度的差异化增益（仅粘度模型）",
            "formula": r"gate_{cyclic} = \mathbb{1}_{salt} \cdot c_{gate} \times f_{cyclic}",
        },
    }
    
    for layer_name, info in layers.items():
        with st.expander(layer_name, expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**特征列表**")
                for f in info["features"]:
                    st.code(f, language=None)
            
            with col2:
                st.markdown(f"**说明**: {info['description']}")
                st.latex(info["formula"])
    
    # 交互式计算器
    st.header("🎮 交互式特征探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入参数")
        
        frac_EC = st.slider("EC 摩尔分数", 0.0, 1.0, 0.3, 0.05)
        frac_DMC = st.slider("DMC 摩尔分数", 0.0, 1.0 - frac_EC, 0.7, 0.05)
        frac_other = 1.0 - frac_EC - frac_DMC
        
        st.caption(f"其他溶剂: {frac_other:.2f}")
        
        T_K = st.slider("温度 (K)", 253.15, 333.15, 298.15, 5.0)
    
    with col2:
        st.subheader("计算结果")
        
        # 简化计算
        eps_EC, eps_DMC = 89.78, 3.10
        eta_EC, eta_DMC = 1.90, 0.59
        
        eps_mix = frac_EC * eps_EC + frac_DMC * eps_DMC
        ln_eta_mix = frac_EC * np.log(eta_EC) + frac_DMC * np.log(eta_DMC)
        synergy = frac_EC * frac_DMC
        
        st.metric("ε_mix (混合介电常数)", f"{eps_mix:.2f}")
        st.metric("η_mix (混合粘度, 25°C)", f"{np.exp(ln_eta_mix):.3f} mPa·s")
        st.metric("synergy_carb (协同效应)", f"{synergy:.4f}")
        st.metric("ε/T", f"{eps_mix/T_K:.4f}")
    
    # 物理意义说明
    st.header("📚 物理意义")
    
    st.markdown("""
    ### 为什么需要协同效应特征？
    
    在锂电池电解液中，电导率 κ 取决于两个关键因素：
    
    1. **离子解离度** ∝ 介电常数 ε
       - 高 ε 溶剂（如 EC, ε=89.8）促进 LiPF6 解离
       
    2. **离子迁移率** ∝ 1/粘度 η
       - 低 η 溶剂（如 DMC, η=0.59 mPa·s）让离子移动更快
    
    当 EC 和 DMC 混合时，兼顾了两者优势，电导率出现**非线性峰值**：
    
    """)
    
    st.latex(r"\kappa_{max} \approx 10-12 \text{ mS/cm at EC:DMC} \approx 30:70")
    
    st.markdown("""
    `synergy_carb = frac_cyclic × frac_linear` 这个简单的乘积特征，
    在 EC=30%, DMC=70% 时达到最大值 0.21，恰好捕捉了这一物理现象。
    
    ### 为什么粘度模型需要门控特征？
    
    盐对粘度的影响在不同溶剂体系中差异显著：
    
    | 溶剂体系 | 1M LiPF6 粘度增益 |
    |----------|------------------|
    | EC/DMC (高极性) | +100% ~ +150% |
    | EP/EA (低极性) | +20% ~ +40% |
    
    通过门控机制 `salt × frac_cyclic` 和 `salt × frac_linear × dipole_norm`，
    模型可以学习这种差异化的盐效应。
    """)


if __name__ == "__main__":
    main()
