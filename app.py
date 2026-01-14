# -*- coding: utf-8 -*-
"""
app.py - 电解液配方性能预测系统 主入口

基于 Streamlit 的 Web 应用
"""

import streamlit as st
import os

# 页面配置
st.set_page_config(
    page_title="电解液性能预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # 标题
    st.markdown('<div class="main-header">🔋 电解液配方性能预测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Electrolyte Performance Prediction System</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 功能介绍
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚡ 电导率预测")
        st.markdown("""
        - 基于 CatBoost 机器学习模型
        - 盐伪装 + 物理修正策略
        - 支持 7 种常见锂盐
        - 预测精度 R² > 0.95
        """)
        
    with col2:
        st.markdown("### 🌊 粘度预测")
        st.markdown("""
        - 残差模式 + Arrhenius 基线
        - 门控盐效应特征工程
        - 支持宽温域预测
        - 预测精度 R² > 0.92
        """)
    
    st.markdown("---")
    
    # 快速导航
    st.markdown("### 🚀 快速开始")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 配方设计", use_container_width=True):
            st.switch_page("pages/1_📊_配方设计.py")
    
    with col2:
        if st.button("🔬 特征原理", use_container_width=True):
            st.switch_page("pages/2_🔬_特征原理.py")
    
    with col3:
        if st.button("📚 溶剂库", use_container_width=True):
            st.switch_page("pages/3_📚_溶剂库.py")
    
    with col4:
        if st.button("ℹ️ 关于", use_container_width=True):
            st.switch_page("pages/4_ℹ️_关于.py")
    
    st.markdown("---")
    
    # 示例配方
    st.markdown("### 📝 示例配方")
    
    example_formulas = {
        "EC:DMC 3:7": "经典配方，平衡电导率和粘度",
        "EC:EMC 3:7": "低温性能较好",
        "EC:EP 3:7": "低粘度配方，适合快充",
        "EC:DMC:EMC 1:1:1": "三元体系，综合性能",
    }
    
    cols = st.columns(len(example_formulas))
    for i, (formula, desc) in enumerate(example_formulas.items()):
        with cols[i]:
            st.markdown(f"**{formula}**")
            st.caption(desc)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Powered by CatBoost & Streamlit | 
        特征工程基于 solvent_mixture_features v4.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
