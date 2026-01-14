# -*- coding: utf-8 -*-
"""
关于模型页面
"""

import streamlit as st

st.set_page_config(
    page_title="关于 - 电解液预测",
    page_icon="ℹ️",
    layout="wide",
)


def main():
    st.title("ℹ️ 关于本系统")
    
    st.markdown("""
    ## 电解液配方性能预测系统
    
    基于机器学习的锂电池电解液电导率和粘度预测工具。
    
    ### 🎯 功能特点
    
    - **电导率预测**: 使用盐伪装 + 物理修正策略，解决训练数据盐种类不均衡问题
    - **粘度预测**: 采用残差模式 + Arrhenius 基线，提供物理可解释的预测
    - **特征工程**: 基于电解液物理化学原理的多层特征设计
    - **交互界面**: 友好的 Web 界面，支持单次和批量预测
    
    ### 📊 模型性能
    
    | 模型 | 训练集 R² | 测试集 R² | 测试集 RMSE |
    |------|----------|----------|-------------|
    | 电导率 | ~0.98 | ~0.95 | ~0.15 (ln-scale) |
    | 粘度 | ~0.96 | ~0.92 | ~0.12 (ln-scale) |

    
    ### 📚 参考文献
    
    1. CALiSol-23 电解液数据库
    2. Gutmann Donor/Acceptor Number 理论
    3. Arrhenius 粘度方程
    4. VTF (Vogel-Tammann-Fulcher) 方程
    
    ### ⚠️ 使用限制
    
    - 温度范围: 暂不支持过高或过低
    - 浓度范围: 0 ~ 5 mol/L
    - 支持的盐: LiPF6, LiFSI, LiTFSI, LiBF4, LiBOB, LiDFOB, LiClO4
    - 溶剂: 需在数据库中有物性数据
    
    ---
    
    **版本**: v1.0  
    **最后更新**: 2026-01-14
    """)


if __name__ == "__main__":
    main()
