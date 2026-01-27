# -*- coding: utf-8 -*-
"""
配方设计与预测页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ElectrolytePredictor,
    SolventDatabase,
    mass_to_mole_fraction,
    normalize_fractions,
    validate_formula,
    validate_conditions,
    get_preset_formula,
    list_preset_formulas,
    get_salt_list,
    get_salt_info,
)

# 页面配置
st.set_page_config(
    page_title="配方设计 - 电解液预测",
    page_icon="📊",
    layout="wide",
)

# 初始化
@st.cache_resource
def load_predictor():
    """加载预测器（缓存）"""
    # 模型路径（根据实际情况修改）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    cond_path = os.path.join(models_dir, "conductivity_catboost.pkl")
    visc_path = os.path.join(models_dir, "viscosity_catboost.pkl")
    
    # 检查模型是否存在
    cond_exists = os.path.exists(cond_path)
    visc_exists = os.path.exists(visc_path)
    
    if not cond_exists and not visc_exists:
        return None, "未找到模型文件，请将模型放入 models/ 目录"
    
    predictor = ElectrolytePredictor(
        conductivity_model_path=cond_path if cond_exists else None,
        viscosity_model_path=visc_path if visc_exists else None,
    )
    
    msg = []
    if cond_exists:
        msg.append("✓ 电导率模型已加载")
    else:
        msg.append("✗ 电导率模型未找到")
    if visc_exists:
        msg.append("✓ 粘度模型已加载")
    else:
        msg.append("✗ 粘度模型未找到")
    
    return predictor, " | ".join(msg)


@st.cache_resource
def load_solvent_db():
    """加载溶剂数据库"""
    return SolventDatabase()


def main():
    st.title("📊 配方设计与预测")
    
    # 加载资源
    predictor, load_msg = load_predictor()
    solvent_db = load_solvent_db()
    
    # 显示模型状态
    if predictor is None:
        st.error(load_msg)
        st.info("请将训练好的模型文件放入 `models/` 目录：\n- `conductivity_catboost.pkl`\n- `viscosity_catboost.pkl`")
        return
    else:
        st.caption(load_msg)
    
    # ========== 侧边栏：实验条件 ==========
    with st.sidebar:
        st.header("⚙️ 实验条件")
        
        # 温度
        st.subheader("🌡️ 温度")
        temp_unit = st.radio("单位", ["°C", "K"], horizontal=True, key="temp_unit")
        
        if temp_unit == "°C":
            temp_c = st.slider("温度 (°C)", min_value=-40, max_value=80, value=25, step=5)
            T_K = temp_c + 273.15
        else:
            T_K = st.slider("温度 (K)", min_value=233.15, max_value=353.15, value=298.15, step=5.0)
            temp_c = T_K - 273.15
        
        st.caption(f"当前: {temp_c:.1f}°C = {T_K:.2f} K")
        
        st.markdown("---")
        
        # 盐配置
        st.subheader("🧂 盐配置")
        
        salt_list = get_salt_list()
        salt = st.selectbox(
            "盐种类",
            options=salt_list,
            index=salt_list.index("LiPF6"),
            format_func=lambda x: f"{x} ({get_salt_info(x).get('name_cn', '')})" if x != "NONE" else "无盐"
        )
        
        # 盐信息提示
        salt_info = get_salt_info(salt)
        if salt_info:
            st.caption(salt_info.get("description", ""))
            st.caption(f"电导率修正系数: {salt_info.get('conductivity_factor', 1.0):.2f}")
        
        # 浓度
        if salt != "NONE":
            col1, col2 = st.columns([2, 1])
            with col1:
                conc = st.number_input("浓度", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            with col2:
                conc_unit = st.selectbox("单位", ["mol/L", "mol/kg"], label_visibility="collapsed")
        else:
            conc = 0.0
            conc_unit = "mol/L"
        
        st.markdown("---")
        
        # 预测按钮
        predict_btn = st.button("🔮 开始预测", use_container_width=True, type="primary")
    
    # ========== 主面板 ==========
    
    # 标签页
    tab1, tab2, tab3 = st.tabs(["🧪 配方输入", "📈 预测结果", "📁 批量预测"])
    
    # ----- Tab 1: 配方输入 -----
    with tab1:
        st.subheader("选择溶剂组分")
        
        # 预设配方
        col1, col2 = st.columns([1, 3])
        with col1:
            preset_options = ["自定义"] + list_preset_formulas()
            preset = st.selectbox("预设配方", preset_options)
        
        # 初始化 session state
        if "formula_solvents" not in st.session_state:
            st.session_state.formula_solvents = ["EC", "DMC"]
            st.session_state.formula_masses = [30.0, 70.0]
        
        # 加载预设
        if preset != "自定义":
            preset_formula = get_preset_formula(preset)
            if preset_formula:
                st.session_state.formula_solvents = list(preset_formula.keys())
                # 转换为质量比（近似）
                st.session_state.formula_masses = [v * 100 for v in preset_formula.values()]
        
        # 输入模式选择
        input_mode = st.radio("输入模式", ["质量比 (%)", "摩尔分数"], horizontal=True)
        
        # 溶剂列表
        all_solvents = solvent_db.get_solvent_list()
        
        # 动态溶剂输入
        st.markdown("##### 溶剂组分")
        
        # 添加溶剂按钮
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➕ 添加溶剂"):
                # 找一个未使用的溶剂
                used = set(st.session_state.formula_solvents)
                for s in all_solvents:
                    if s not in used:
                        st.session_state.formula_solvents.append(s)
                        st.session_state.formula_masses.append(0.0)
                        break
                st.rerun()
        
        # 显示溶剂输入行
        formula_data = []
        total_mass = 0.0
        
        for i, (solvent, mass) in enumerate(zip(
            st.session_state.formula_solvents,
            st.session_state.formula_masses
        )):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 0.5])
            
            with col1:
                new_solvent = st.selectbox(
                    f"溶剂 {i+1}",
                    options=all_solvents,
                    index=all_solvents.index(solvent) if solvent in all_solvents else 0,
                    key=f"solvent_{i}",
                    label_visibility="collapsed",
                )
                st.session_state.formula_solvents[i] = new_solvent
            
            with col2:
                if input_mode == "质量比 (%)":
                    new_mass = st.number_input(
                        "质量比",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(mass),
                        step=5.0,
                        key=f"mass_{i}",
                        label_visibility="collapsed",
                    )
                else:
                    new_mass = st.number_input(
                        "摩尔分数",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(mass) / 100 if mass <= 1 else float(mass) / 100,
                        step=0.05,
                        key=f"mass_{i}",
                        label_visibility="collapsed",
                    )
                    new_mass = new_mass * 100  # 内部用百分比存储
                st.session_state.formula_masses[i] = new_mass
            
            with col3:
                props = solvent_db.get_properties(new_solvent)
                if props:
                    st.caption(props.get("name_cn", ""))
            
            with col4:
                if len(st.session_state.formula_solvents) > 1:
                    if st.button("✕", key=f"del_{i}"):
                        st.session_state.formula_solvents.pop(i)
                        st.session_state.formula_masses.pop(i)
                        st.rerun()
            
            if new_mass > 0:
                formula_data.append((new_solvent, new_mass))
                total_mass += new_mass
        
        # 显示总和
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if input_mode == "质量比 (%)":
                color = "green" if abs(total_mass - 100) < 0.1 else "red"
                st.markdown(f"**总计**: :{color}[{total_mass:.1f}%]")
            else:
                total_frac = total_mass / 100
                color = "green" if abs(total_frac - 1.0) < 0.01 else "red"
                st.markdown(f"**总计**: :{color}[{total_frac:.3f}]")
        
        with col2:
            st.caption("💡 输入会自动归一化")
    
    # ----- Tab 2: 预测结果 -----
    with tab2:
        if predict_btn or st.session_state.get("last_prediction"):
            # 验证条件
            is_valid, err_msg = validate_conditions(T_K, conc, salt, conc_unit)
            if not is_valid:
                st.error(f"条件验证失败: {err_msg}")
            else:
                # 构建配方
                formula_dict = {}
                for solvent, mass in formula_data:
                    if solvent in formula_dict:
                        formula_dict[solvent] += mass
                    else:
                        formula_dict[solvent] = mass
                
                # 归一化
                if input_mode == "质量比 (%)":
                    try:
                        mole_frac = mass_to_mole_fraction(formula_dict)
                    except Exception as e:
                        st.error(f"质量比转换失败: {e}")
                        mole_frac = normalize_fractions(formula_dict)
                else:
                    mole_frac = normalize_fractions({k: v/100 for k, v in formula_dict.items()})
                
                # 预测
                with st.spinner("预测中..."):
                    result = predictor.predict_both(
                        formula=mole_frac,
                        T_K=T_K,
                        salt=salt,
                        conc=conc,
                        conc_unit=conc_unit,
                    )
                
                st.session_state.last_prediction = result
                
                # 显示结果
                st.subheader("📊 预测结果")
                
                col1, col2 = st.columns(2)
                
                # 电导率
                with col1:
                    cond = result["conductivity"]
                    if cond["success"]:
                        st.metric(
                            label="⚡ 电导率",
                            value=f"{cond['k_pred_final']:.2f} mS/cm",
                        )
                        st.caption(f"LiPF6 当量: {cond['k_pred_base']:.2f} mS/cm")
                        st.caption(f"盐修正系数: {cond['salt_correction']:.2f}")
                    else:
                        st.error(f"电导率预测失败: {cond['error']}")
                
                # 粘度
                with col2:
                    visc = result["viscosity"]
                    if visc["success"]:
                        st.metric(
                            label="🌊 粘度",
                            value=f"{visc['eta_pred']:.2f} mPa·s",
                        )
                        st.caption(f"Arrhenius 基线: {np.exp(visc['ln_eta_ideal']):.2f} mPa·s")
                        st.caption(f"盐效应残差: {visc['residual']:+.3f}")
                    else:
                        st.error(f"粘度预测失败: {visc['error']}")
                
                # 配方详情
                st.markdown("---")
                st.subheader("📋 配方详情")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**溶剂组成 (摩尔分数)**")
                    for solvent, frac in mole_frac.items():
                        st.write(f"- {solvent}: {frac:.3f}")
                
                with col2:
                    st.markdown("**实验条件**")
                    st.write(f"- 温度: {temp_c:.1f}°C ({T_K:.2f} K)")
                    st.write(f"- 盐: {salt}")
                    if salt != "NONE":
                        st.write(f"- 浓度: {conc} {conc_unit}")
                
                  # ========== 下载功能 ==========
                st.markdown("---")
                st.subheader("📥 导出预测结果")
                
                # 构建下载数据
                download_data = {
                    "温度_C": temp_c,
                    "温度_K": T_K,
                    "盐种类": salt,
                    "盐浓度": conc,
                    "浓度单位": conc_unit,
                }
                
                # 添加溶剂配方（摩尔分数）
                for solvent, frac in mole_frac.items():
                    download_data[f"{solvent}_摩尔分数"] = round(frac, 4)
                
                # 添加质量比
                total = sum(formula_dict.values())
                for solvent, mass in formula_dict.items():
                    download_data[f"{solvent}_质量百分比"] = round(mass / total * 100, 2)
                
                # 添加预测结果（只保留核心数据）
                cond = result["conductivity"]
                visc = result["viscosity"]
                
                if cond["success"]:
                    download_data["电导率_mS_cm"] = round(cond['k_pred_final'], 4)
                else:
                    download_data["电导率_mS_cm"] = "预测失败"
                
                if visc["success"]:
                    download_data["粘度_mPa_s"] = round(visc['eta_pred'], 4)
                else:
                    download_data["粘度_mPa_s"] = "预测失败"
                
                # 下载按钮
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV 下载
                    csv_df = pd.DataFrame([download_data])
                    csv_data = csv_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📄 下载 CSV",
                        data=csv_data,
                        file_name=f"prediction_{salt}_{temp_c}C.csv",
                        mime="text/csv",
                    )
                
                with col2:
                    # JSON 下载
                    import json
                    json_data = json.dumps(download_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📋 下载 JSON",
                        data=json_data.encode('utf-8'),
                        file_name=f"prediction_{salt}_{temp_c}C.json",
                        mime="application/json",
                    )
                
                with col3:
                    # 复制到剪贴板的文本格式
                    text_lines = [
                        "=" * 40,
                        "电解液配方预测结果",
                        "=" * 40,
                        "",
                        "【实验条件】",
                        f"  温度: {temp_c:.1f}°C ({T_K:.2f} K)",
                        f"  盐: {salt}",
                        f"  浓度: {conc} {conc_unit}",
                        "",
                        "【溶剂配方 (摩尔分数)】",
                    ]
                    for solvent, frac in mole_frac.items():
                        text_lines.append(f"  {solvent}: {frac:.4f}")
                    
                    text_lines.extend([
                        "",
                        "【预测结果】",
                        f"  电导率: {cond['k_pred_final']:.2f} mS/cm" if cond["success"] else "  电导率: 预测失败",
                        f"  粘度: {visc['eta_pred']:.2f} mPa·s" if visc["success"] else "  粘度: 预测失败",
                        "",
                        "=" * 40,
                    ])
                    text_report = "\n".join(text_lines)
                    
                    st.download_button(
                        label="📝 下载报告",
                        data=text_report.encode('utf-8'),
                        file_name=f"prediction_{salt}_{temp_c}C.txt",
                        mime="text/plain",
                    )
                
                # 显示预览
                with st.expander("👀 预览下载内容"):
                    st.json(download_data)
                
        else:
            st.info("👈 请在侧边栏设置条件，然后点击「开始预测」")
    
    # ----- Tab 3: 批量预测 -----
    with tab3:
        st.subheader("📁 批量预测")
        
        st.markdown("""
        上传 CSV 文件进行批量预测。文件需包含以下列：
        - `T_K`: 温度 (K)
        - `salt_clean`: 盐种类
        - `c_value`: 浓度
        - `c_unit`: 浓度单位
        - `frac_EC`, `frac_DMC`, ...: 溶剂摩尔分数
        """)
        
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"已加载 {len(df)} 条数据")
            st.dataframe(df.head())
            
            if st.button("🚀 开始批量预测"):
                with st.spinner("批量预测中..."):
                    result_df = predictor.batch_predict(df)
                
                st.success("预测完成！")
                st.dataframe(result_df)
                
                # 下载按钮
                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="下载结果",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
