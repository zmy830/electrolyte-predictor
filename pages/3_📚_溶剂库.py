# -*- coding: utf-8 -*-
"""
溶剂数据库管理页面
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SolventDatabase, SALT_DATABASE, get_salt_list

st.set_page_config(
    page_title="溶剂库 - 电解液预测",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_db():
    return SolventDatabase()


def main():
    st.title("📚 溶剂物性数据库")
    
    db = load_db()
    
    # 标签页
    tab1, tab2 = st.tabs(["🧪 溶剂库", "🧂 盐库"])
    
    # ----- 溶剂库 -----
    with tab1:
        # 筛选
        col1, col2 = st.columns([1, 4])
        with col1:
            category = st.selectbox(
                "类型筛选",
                ["全部", "cyclic", "linear", "ether", "other"],
            )
        
        # 获取数据
        if category == "全部":
            solvents = db.get_solvent_list()
        else:
            solvents = db.get_solvent_list(category=category)
        
        # 转换为 DataFrame
        data = []
        for name in solvents:
            props = db.get_properties(name)
            if props:
                data.append({
                    "名称": name,
                    "中文名": props.get("name_cn", ""),
                    "类型": props.get("type", ""),
                    "ε (25°C)": props.get("eps_r", ""),
                    "η (mPa·s)": props.get("eta_25", ""),
                    "DN": props.get("DN", ""),
                    "AN": props.get("AN", ""),
                    "MW": props.get("MW", ""),
                    "ρ (g/cm³)": props.get("rho", ""),
                    "μ (D)": props.get("dipole", ""),
                })
        
        df = pd.DataFrame(data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ε (25°C)": st.column_config.NumberColumn(format="%.2f"),
                "η (mPa·s)": st.column_config.NumberColumn(format="%.2f"),
                "MW": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        
        st.caption(f"共 {len(solvents)} 种溶剂")
        
        # 下载按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出 CSV",
            data=csv,
            file_name="solvent_database.csv",
            mime="text/csv",
        )
    
    # ----- 盐库 -----
    with tab2:
        st.subheader("锂盐数据库")
        
        salt_data = []
        for name, info in SALT_DATABASE.items():
            salt_data.append({
                "名称": name,
                "中文名": info.get("name_cn", ""),
                "分子量": info.get("MW", 0),
                "电导率修正系数": info.get("conductivity_factor", 1.0),
                "说明": info.get("description", ""),
            })
        
        salt_df = pd.DataFrame(salt_data)
        
        st.dataframe(
            salt_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "电导率修正系数": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        
        st.markdown("""
        **电导率修正系数说明**：
        - 以 LiPF6 为基准 (1.00)
        - > 1.0 表示电导率高于 LiPF6
        - < 1.0 表示电导率低于 LiPF6
        """)


if __name__ == "__main__":
    main()
