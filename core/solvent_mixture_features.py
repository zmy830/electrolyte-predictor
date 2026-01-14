#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
solvent_mixture_features.py (v4.0 统一版)

功能：
1. 定义锂电池溶剂物性表（含分类、偶极矩、温度锚点）
2. 提供 add_mixture_features() 函数，为电导率/粘度模型计算混合物特征

特征分层设计：
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0 - 物性底座 (Base Properties)                            │
│   eps_mix, ln_eta0_mix, DN_mix, AN_mix, MW_mix, rho_mix        │
│   dipole_mix (粘度专用)                                         │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1 - 结构分组 (Structure Fractions)                        │
│   frac_cyclic, frac_linear, frac_ether                          │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2 - 协同效应 (Synergy)                                    │
│   synergy_carb = frac_cyclic × frac_linear                      │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3 - 温度交互 (Temperature Coupling)                       │
│   eps_over_T, inv_T, inv_TmT0                                   │
│   ln_eta_ideal_T (粘度专用，Arrhenius基线)                       │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4 - 盐效应门控 (Salt Effect, 粘度专用)                     │
│   通过训练脚本的 add_salt_gates() 添加                           │
└─────────────────────────────────────────────────────────────────┘

作者：Claude (Anthropic)
版本：4.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

# ============================================================================
# 1. 溶剂物性表
# ============================================================================
# type 定义:
#   cyclic: 高介电、高粘度 (EC, PC, FEC, GBL) -> 提供解离能力
#   linear: 低介电、低粘度 (DMC, EMC, DEC, EA, MA, EP, PP) -> 提供流动性
#   ether:  醚类 (DME, DOL, THF, Glymes) -> 强溶剂化能力
#   other:  其他 (砜类, 腈类, 芳香烃等)

SOLVENT_PROPS: Dict[str, Dict[str, float]] = {
    # =========================================================================
    # 1. 环状碳酸酯/内酯 (Cyclic) - 高介电、高粘度，提供离子解离能力
    # =========================================================================
    "EC":   {"eps_r": 89.78, "eta_25": 1.90, "DN": 16.4, "AN": 18.0, "MW": 88.06,  "rho": 1.32, "type": "cyclic"},
    "PC":   {"eps_r": 64.92, "eta_25": 2.53, "DN": 15.1, "AN": 18.3, "MW": 102.09, "rho": 1.20, "type": "cyclic"},
    "FEC":  {"eps_r": 102.0, "eta_25": 4.10, "DN": 14.0, "AN": 20.0, "MW": 106.05, "rho": 1.45, "type": "cyclic"},
    "GBL":  {"eps_r": 39.0,  "eta_25": 1.70, "DN": 18.0, "AN": 18.0, "MW": 86.09,  "rho": 1.12, "type": "cyclic"},  # g-Butyrolactone
    "VC":   {"eps_r": 126.0, "eta_25": 3.50, "DN": 15.0, "AN": 19.0, "MW": 86.05,  "rho": 1.36, "type": "cyclic"},  # Vinylene Carbonate
    "3-Me-2-Oxazolidinone": {"eps_r": 77.5, "eta_25": 2.50, "DN": 25.0, "AN": 15.0, "MW": 101.10, "rho": 1.17, "type": "cyclic"},

    # =========================================================================
    # 2. 链状碳酸酯/羧酸酯 (Linear) - 低介电、低粘度，提供流动性
    # =========================================================================
    "DMC":  {"eps_r": 3.10,  "eta_25": 0.59, "DN": 15.1, "AN": 14.0, "MW": 90.08,  "rho": 1.07, "type": "linear"},
    "EMC":  {"eps_r": 2.96,  "eta_25": 0.65, "DN": 16.0, "AN": 14.0, "MW": 104.10, "rho": 1.01, "type": "linear"},
    "DEC":  {"eps_r": 2.80,  "eta_25": 0.75, "DN": 16.0, "AN": 14.0, "MW": 118.13, "rho": 0.97, "type": "linear"},
    "EA":   {"eps_r": 6.02,  "eta_25": 0.43, "DN": 17.1, "AN": 14.0, "MW": 88.11,  "rho": 0.90, "type": "linear"},  # Ethyl Acetate
    "MA":   {"eps_r": 6.68,  "eta_25": 0.36, "DN": 16.5, "AN": 14.0, "MW": 74.08,  "rho": 0.93, "type": "linear"},  # Methyl Acetate
    "EP":   {"eps_r": 5.60,  "eta_25": 0.52, "DN": 17.0, "AN": 12.0, "MW": 102.13, "rho": 0.89, "type": "linear"},  # Ethyl Propionate
    "PP":   {"eps_r": 4.70,  "eta_25": 0.65, "DN": 17.0, "AN": 12.0, "MW": 116.16, "rho": 0.88, "type": "linear"},  # Propyl Propionate
    "MF":   {"eps_r": 8.50,  "eta_25": 0.33, "DN": 17.0, "AN": 15.0, "MW": 60.05,  "rho": 0.97, "type": "linear"},  # Methyl Formate
    "MOEMC":{"eps_r": 7.00,  "eta_25": 1.10, "DN": 17.0, "AN": 14.0, "MW": 134.13, "rho": 1.10, "type": "linear"},

    # =========================================================================
    # 3. 醚类 (Ether) - 中等介电、低粘度，强溶剂化能力
    # =========================================================================
    "DME":  {"eps_r": 7.20,  "eta_25": 0.46, "DN": 20.0, "AN": 10.0, "MW": 90.12,  "rho": 0.87, "type": "ether"},
    "DOL":  {"eps_r": 7.10,  "eta_25": 0.58, "DN": 18.0, "AN": 10.0, "MW": 74.08,  "rho": 1.06, "type": "ether"},  # 1,3-Dioxolane
    "THF":  {"eps_r": 7.58,  "eta_25": 0.46, "DN": 20.0, "AN": 8.0,  "MW": 72.11,  "rho": 0.89, "type": "ether"},
    "2-MeTHF": {"eps_r": 6.20, "eta_25": 0.46, "DN": 18.0, "AN": 8.0, "MW": 86.13, "rho": 0.85, "type": "ether"},
    "DMM":  {"eps_r": 2.65,  "eta_25": 0.32, "DN": 17.0, "AN": 8.0,  "MW": 76.09,  "rho": 0.86, "type": "ether"},  # Dimethoxymethane
    "Diglyme":      {"eps_r": 7.23, "eta_25": 0.98, "DN": 19.5, "AN": 10.0, "MW": 134.17, "rho": 0.94, "type": "ether"},  # 2-Glyme
    "Triglyme":     {"eps_r": 7.50, "eta_25": 1.96, "DN": 18.0, "AN": 10.0, "MW": 178.23, "rho": 0.98, "type": "ether"},  # 3-Glyme
    "Tetraglyme":   {"eps_r": 7.70, "eta_25": 4.05, "DN": 16.6, "AN": 10.0, "MW": 222.28, "rho": 1.01, "type": "ether"},  # 4-Glyme
    "Ethylmonoglyme": {"eps_r": 5.10, "eta_25": 0.60, "DN": 19.0, "AN": 10.0, "MW": 118.17, "rho": 0.84, "type": "ether"},
    "Ethyldiglyme":   {"eps_r": 5.70, "eta_25": 1.24, "DN": 18.5, "AN": 10.0, "MW": 162.23, "rho": 0.91, "type": "ether"},

    # =========================================================================
    # 4. 砜类 (Sulfones) - 高热稳定性
    # =========================================================================
    "Sulfolane":    {"eps_r": 43.3,  "eta_25": 10.3, "DN": 14.8, "AN": 19.0, "MW": 120.17, "rho": 1.26, "type": "other"},
    "3-MeSulfolane":{"eps_r": 29.0,  "eta_25": 6.00, "DN": 14.5, "AN": 18.0, "MW": 134.20, "rho": 1.18, "type": "other"},
    "Propylsulfone":{"eps_r": 25.0,  "eta_25": 5.00, "DN": 15.0, "AN": 18.0, "MW": 150.24, "rho": 1.10, "type": "other"},

    # =========================================================================
    # 5. 极性非质子溶剂 (Polar Aprotic)
    # =========================================================================
    "DMSO": {"eps_r": 46.7,  "eta_25": 1.99, "DN": 29.8, "AN": 19.3, "MW": 78.13,  "rho": 1.10, "type": "other"},
    "AN":   {"eps_r": 37.5,  "eta_25": 0.34, "DN": 14.1, "AN": 18.9, "MW": 41.05,  "rho": 0.78, "type": "other"},  # Acetonitrile
    "DMF":  {"eps_r": 36.7,  "eta_25": 0.80, "DN": 26.6, "AN": 16.0, "MW": 73.09,  "rho": 0.94, "type": "other"},

    # =========================================================================
    # 6. 氟代溶剂 (Fluorinated)
    # =========================================================================
    "TFP":  {"eps_r": 6.00,  "eta_25": 4.50, "DN": 12.0, "AN": 10.0, "MW": 344.07, "rho": 1.55, "type": "other"},

    # =========================================================================
    # 7. 芳香烃 (Aromatics) - 非极性稀释剂
    # =========================================================================
    "Toluene":      {"eps_r": 2.38, "eta_25": 0.56, "DN": 0.1, "AN": 0.1, "MW": 92.14,  "rho": 0.87, "type": "other"},
    "Benzene":      {"eps_r": 2.27, "eta_25": 0.60, "DN": 0.1, "AN": 0.1, "MW": 78.11,  "rho": 0.87, "type": "other"},
    "Ethylbenzene": {"eps_r": 2.40, "eta_25": 0.63, "DN": 0.1, "AN": 0.1, "MW": 106.17, "rho": 0.87, "type": "other"},
    "Cumene":       {"eps_r": 2.38, "eta_25": 0.70, "DN": 0.1, "AN": 0.1, "MW": 120.19, "rho": 0.86, "type": "other"},  # Isopropylbenzene
    "Pseudocumene": {"eps_r": 2.37, "eta_25": 0.80, "DN": 0.1, "AN": 0.1, "MW": 120.19, "rho": 0.88, "type": "other"},  # 1,2,4-Trimethylbenzene
    "m-Xylene":     {"eps_r": 2.37, "eta_25": 0.58, "DN": 0.1, "AN": 0.1, "MW": 106.17, "rho": 0.86, "type": "other"},
    "o-Xylene":     {"eps_r": 2.57, "eta_25": 0.76, "DN": 0.1, "AN": 0.1, "MW": 106.17, "rho": 0.88, "type": "other"},

    # =========================================================================
    # 8. 卤代烃 & 其他 (Halogenated & Others)
    # =========================================================================
    "DCM":          {"eps_r": 8.93,  "eta_25": 0.41, "DN": 1.0,  "AN": 20.0, "MW": 84.93,  "rho": 1.33, "type": "other"},  # Methylene chloride
    "Freon11":      {"eps_r": 2.28,  "eta_25": 0.42, "DN": 0.0,  "AN": 0.0,  "MW": 137.37, "rho": 1.49, "type": "other"},  # Trichlorofluoromethane
    "TEOS":         {"eps_r": 4.00,  "eta_25": 0.70, "DN": 10.0, "AN": 5.0,  "MW": 208.33, "rho": 0.93, "type": "other"},  # Tetraethyl orthosilicate
}

# 别名映射（处理数据中不同命名）
SOLVENT_ALIASES: Dict[str, str] = {
    # 内酯
    "g-Butyrolactone": "GBL",
    "γ-Butyrolactone": "GBL",
    "gamma-Butyrolactone": "GBL",
    # Glymes
    "2-Glyme": "Diglyme",
    "3-Glyme": "Triglyme", 
    "4-Glyme": "Tetraglyme",
    # 卤代烃
    "Methylene chloride": "DCM",
    "Dichloromethane": "DCM",
    "Freon 11": "Freon11",
    # 其他
    "Acetonitrile": "AN",
    "Pseudocumeme": "Pseudocumene",  # 拼写修正
}

# ============================================================================
# 2. 偶极矩表 (Debye) - 用于粘度模型
# ============================================================================
DIPOLE_D: Dict[str, float] = {
    # 环状碳酸酯
    "EC": 5.35, "PC": 4.90, "FEC": 5.60, "VC": 5.20, "GBL": 4.12,
    # 链状碳酸酯
    "DMC": 0.90, "EMC": 0.89, "DEC": 0.90,
    # 羧酸酯
    "EA": 1.78, "MA": 1.72, "EP": 1.74, "PP": 1.78, "MF": 1.77,
    # 醚类
    "DME": 1.71, "DOL": 1.90, "THF": 1.75, "2-MeTHF": 1.60, "DMM": 0.90,
    "Diglyme": 1.97, "Triglyme": 2.10, "Tetraglyme": 2.20,
    # 极性非质子
    "AN": 3.92, "DMF": 3.82, "DMSO": 3.96,
    # 砜类
    "Sulfolane": 4.35, "3-MeSulfolane": 4.10,
    # 芳香烃 (非极性)
    "Toluene": 0.36, "Benzene": 0.0, "m-Xylene": 0.30, "o-Xylene": 0.62,
    # 卤代烃
    "DCM": 1.60,
}

# ============================================================================
# 3. 纯溶剂 η(T) 锚点 (mPa·s) - 用于粘度模型 Arrhenius 基线
# ============================================================================
# 格式: {溶剂名: {T_K: η_mPas}}
# 数据来源：DDBST, NIST, 文献实测值
ETA_ANCHOR: Dict[str, Dict[float, float]] = {
    # 环状碳酸酯
    "EC":  {313.15: 1.90},  # EC 低温为固体，仅 40°C 可用
    "PC":  {273.15: 4.27, 298.15: 2.50, 313.15: 1.91},
    "GBL": {273.15: 2.50, 298.15: 1.70, 313.15: 1.35},
    # 链状碳酸酯
    "DMC": {273.15: 0.96, 298.15: 0.59, 313.15: 0.48},
    "EMC": {273.15: 0.91, 298.15: 0.65, 313.15: 0.52},
    "DEC": {273.15: 1.08, 298.15: 0.74, 313.15: 0.60},
    # 羧酸酯
    "EA":  {273.15: 0.68, 298.15: 0.43, 313.15: 0.35},
    "MA":  {273.15: 0.45, 298.15: 0.37, 313.15: 0.33},
    "EP":  {273.15: 0.67, 298.15: 0.49, 313.15: 0.42},
    "PP":  {273.15: 1.28, 298.15: 0.87, 313.15: 0.71},
    "MF":  {293.15: 0.35},  # 仅 20°C 数据
    # 醚类
    "THF": {273.15: 0.57, 298.15: 0.41, 313.15: 0.36},
    "DME": {273.15: 0.60, 298.15: 0.46, 313.15: 0.38},
    "DOL": {273.15: 0.75, 298.15: 0.58, 313.15: 0.48},
    # 极性非质子
    "AN":  {273.15: 0.47, 298.15: 0.37, 313.15: 0.33},
    "DMF": {273.15: 1.10, 298.15: 0.80, 313.15: 0.65},
    "DMSO":{298.15: 1.99, 313.15: 1.50},
    # 砜类
    "Sulfolane": {303.15: 10.3, 313.15: 7.0, 333.15: 4.0},
}


# ============================================================================
# 4. 辅助函数
# ============================================================================

def _fit_arrhenius(points: Dict[float, float]) -> Tuple[float, float]:
    """
    拟合 ln(η) = A + B/T (Arrhenius 形式)
    返回 (A, B)
    """
    Ts = np.array(sorted(points.keys()), dtype=float)
    etas = np.array([points[T] for T in Ts], dtype=float)
    etas = np.clip(etas, 1e-12, None)
    
    if len(Ts) == 1:
        return float(np.log(etas[0])), 0.0
    
    x = 1.0 / Ts
    y = np.log(etas)
    B, A = np.polyfit(x, y, deg=1)
    return float(A), float(B)


# 预计算 Arrhenius 系数
_ETA_ARRHENIUS: Dict[str, Tuple[float, float]] = {
    k: _fit_arrhenius(v) for k, v in ETA_ANCHOR.items()
}


def _ln_eta_at_T(solvent: str, T_K: float, props: Dict) -> float:
    """
    计算溶剂在温度 T 的 ln(η)
    优先使用 Arrhenius 模型，否则退化为 25°C 常数
    """
    if solvent in _ETA_ARRHENIUS and np.isfinite(T_K) and T_K > 100:
        A, B = _ETA_ARRHENIUS[solvent]
        return A + B / T_K
    
    # Fallback: 25°C 常数
    eta25 = props.get(solvent, {}).get("eta_25", 1.0)
    return float(np.log(max(eta25, 1e-12)))


def _resolve_solvent_name(name: str) -> str:
    """解析溶剂名（处理别名）"""
    return SOLVENT_ALIASES.get(name, name)


def get_solvent_table() -> pd.DataFrame:
    """获取溶剂物性表（DataFrame 格式）"""
    df = pd.DataFrame.from_dict(SOLVENT_PROPS, orient="index")
    df.index.name = "solvent"
    return df.reset_index()


# ============================================================================
# 5. 主函数: add_mixture_features
# ============================================================================

def add_mixture_features(
    df: pd.DataFrame,
    frac_prefix: str = "frac_",
    T_col: str = "T_K",
    c_col: str = "c_value",
    task: str = "both",  # "conductivity" | "viscosity" | "both"
) -> pd.DataFrame:
    """
    计算混合物物性特征
    
    Parameters
    ----------
    df : DataFrame
        输入数据，需包含 frac_XXX 列（溶剂摩尔分数）
    frac_prefix : str
        分数列前缀，默认 "frac_"
    T_col : str
        温度列名，默认 "T_K"
    c_col : str
        浓度列名，默认 "c_value"
    task : str
        任务类型：
        - "conductivity": 电导率模型（精简特征）
        - "viscosity": 粘度模型（含 Arrhenius 基线）
        - "both": 两者都计算
    
    Returns
    -------
    DataFrame
        添加了物性特征的数据框
    
    输出特征
    --------
    [共用特征]
    - eps_mix       : 混合介电常数 (25°C)
    - ln_eta0_mix   : 混合 ln(η) at 25°C
    - DN_mix        : 混合 Donor Number
    - AN_mix        : 混合 Acceptor Number
    - MW_mix        : 混合分子量
    - rho_mix       : 混合密度 (25°C)
    - frac_cyclic   : 环状溶剂总分数
    - frac_linear   : 链状溶剂总分数
    - frac_ether    : 醚类溶剂总分数
    - synergy_carb  : 协同效应 = frac_cyclic × frac_linear
    - eps_over_T    : eps_mix / T
    
    [粘度专用 - task="viscosity" 或 "both"]
    - dipole_mix      : 混合偶极矩 (D)
    - ln_eta_ideal_T  : Arrhenius 混合基线 ln(η) at T
    """
    # 获取分数列
    frac_cols = [c for c in df.columns if c.startswith(frac_prefix)]
    if not frac_cols:
        raise ValueError(f"未找到以 '{frac_prefix}' 开头的列")
    
    solvent_names = [c[len(frac_prefix):] for c in frac_cols]
    
    # 检查未知溶剂
    known = set(SOLVENT_PROPS.keys()) | set(SOLVENT_ALIASES.keys())
    unknown = [s for s in solvent_names if _resolve_solvent_name(s) not in SOLVENT_PROPS]
    if unknown:
        print(f"[Warning] 未知溶剂将被忽略: {unknown}")
    
    # 预取偶极矩数组
    dipole_arr = np.array([
        DIPOLE_D.get(_resolve_solvent_name(s), np.nan) 
        for s in solvent_names
    ], dtype=float)
    
    do_visc = task in ("viscosity", "both")
    
    # 计算特征
    results = []
    for idx, row in df.iterrows():
        w_raw = row[frac_cols].values.astype(float)
        total = np.nansum(w_raw)
        
        if not np.isfinite(total) or total <= 0:
            results.append({})
            continue
        
        w = w_raw / total  # 归一化
        T_K = float(row.get(T_col, 298.15))
        
        # 初始化累加器
        eps_sum = 0.0
        ln_eta25_sum = 0.0
        dn_sum = 0.0
        an_sum = 0.0
        mw_sum = 0.0
        rho_sum = 0.0
        ln_eta_T_sum = 0.0
        
        cyc_sum = 0.0
        lin_sum = 0.0
        eth_sum = 0.0
        
        for i, raw_name in enumerate(solvent_names):
            wi = float(w[i])
            if wi < 1e-12:
                continue
            
            name = _resolve_solvent_name(raw_name)
            if name not in SOLVENT_PROPS:
                continue
            
            p = SOLVENT_PROPS[name]
            
            # 基础物性
            eps_sum += wi * p["eps_r"]
            ln_eta25_sum += wi * np.log(max(p["eta_25"], 1e-12))
            dn_sum += wi * p["DN"]
            an_sum += wi * p["AN"]
            mw_sum += wi * p["MW"]
            rho_sum += wi * p["rho"]
            
            # 结构分类
            stype = p.get("type", "other")
            if stype == "cyclic":
                cyc_sum += wi
            elif stype == "linear":
                lin_sum += wi
            elif stype == "ether":
                eth_sum += wi
            
            # Arrhenius 基线 (粘度用)
            if do_visc:
                ln_eta_T_sum += wi * _ln_eta_at_T(name, T_K, SOLVENT_PROPS)
        
        # 偶极矩 (忽略 NaN 重归一化)
        mask = np.isfinite(dipole_arr) & (w > 1e-12)
        if np.any(mask):
            dipole_mix = float(np.sum(w[mask] * dipole_arr[mask]) / np.sum(w[mask]))
        else:
            dipole_mix = np.nan
        
        # 构建特征字典
        feat = {
            # Layer 0: 基础物性
            "eps_mix": eps_sum,
            "ln_eta0_mix": ln_eta25_sum,
            "DN_mix": dn_sum,
            "AN_mix": an_sum,
            "MW_mix": mw_sum,
            "rho_mix": rho_sum,
            
            # Layer 1: 结构分组
            "frac_cyclic": cyc_sum,
            "frac_linear": lin_sum,
            "frac_ether": eth_sum,
            
            # Layer 2: 协同效应
            "synergy_carb": cyc_sum * lin_sum,
            
            # Layer 3: 温度交互
            "eps_over_T": eps_sum / T_K if T_K > 0 else np.nan,
        }
        
        # 粘度专用特征
        if do_visc:
            feat["dipole_mix"] = dipole_mix
            feat["ln_eta_ideal_T"] = ln_eta_T_sum
        
        results.append(feat)
    
    # 合并结果
    feat_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    
    # 去除重复列（保留最后生成的，即新计算的特征）
    out = out.loc[:, ~out.columns.duplicated(keep='last')]
    
    # 添加温度衍生特征
    if T_col in out.columns:
        T = out[T_col].astype(float)
        out["inv_T"] = 1.0 / T
        T0 = 228.0  # VTF 参考温度
        out["inv_TmT0"] = 1.0 / (T - T0)
    
    return out


# ============================================================================
# 6. 粘度专用: 盐效应门控特征
# ============================================================================

def add_salt_gates(
    df: pd.DataFrame,
    c_col: str = "c_value",
    salt_col: str = "salt_clean",
    dip_min: float = None,
    dip_max: float = None,
) -> Tuple[pd.DataFrame, float, float]:
    """
    为粘度模型添加盐效应门控特征
    
    核心思想：
    - 盐对粘度的增益效应在不同溶剂体系中差异显著
    - 高极性环状体系 (EC/PC): 盐效应强
    - 低极性线性体系 (EP/EA/DEC): 盐效应弱
    - 通过门控机制让模型学习这种差异
    
    Parameters
    ----------
    df : DataFrame
        需包含 frac_cyclic, frac_linear, dipole_mix, c_value, salt_clean
    dip_min, dip_max : float
        偶极矩归一化范围（训练时自动计算，预测时需传入训练值）
    
    Returns
    -------
    (df, dip_min, dip_max)
    
    输出特征
    --------
    - has_salt          : 是否有盐 (0/1)
    - c_gate            : log(1 + clip(c, 0, 2)) 压缩浓度
    - dipole_norm       : 偶极矩归一化 [0,1]
    - salt_x_cyclic     : 盐效应 × 环状分数
    - salt_x_linear_hi  : 盐效应 × 线性分数 × 高极性
    - salt_x_linear_lo  : 盐效应 × 线性分数 × 低极性
    """
    # 去除重复列（保留第一个）
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # 基础特征（缺失则补0）
    for col in ["frac_cyclic", "frac_linear"]:
        if col not in df.columns:
            df[col] = 0.0
    
    # 浓度处理
    c = pd.to_numeric(df.get(c_col, 0), errors="coerce").fillna(0.0)
    c_clip = np.clip(c, 0.0, 2.0)
    df["c_gate"] = np.log1p(c_clip)
    
    # 盐标记
    salt = df.get(salt_col, "NONE").astype(str)
    df["has_salt"] = (salt != "NONE").astype(int)
    
    # 偶极矩归一化
    dip = pd.to_numeric(df.get("dipole_mix", np.nan), errors="coerce")
    if dip_min is None or dip_max is None:
        dip_min = float(np.nanmin(dip.values))
        dip_max = float(np.nanmax(dip.values))
        if not np.isfinite(dip_min) or not np.isfinite(dip_max) or dip_max <= dip_min:
            dip_min, dip_max = 0.0, 5.0  # 默认范围
    
    dip_norm = (dip - dip_min) / (dip_max - dip_min + 1e-12)
    df["dipole_norm"] = np.clip(dip_norm, 0.0, 1.0).fillna(0.5)
    
    # 门控特征 - 使用 .values 避免索引对齐问题
    has_salt = df["has_salt"].values
    c_gate = df["c_gate"].values
    frac_cyclic = df["frac_cyclic"].values
    frac_linear = df["frac_linear"].values
    dipole_norm = df["dipole_norm"].values
    
    salt_eff = has_salt * c_gate
    df["salt_x_cyclic"] = salt_eff * frac_cyclic
    df["salt_x_linear_hi"] = salt_eff * frac_linear * dipole_norm
    df["salt_x_linear_lo"] = salt_eff * frac_linear * (1.0 - dipole_norm)
    
    return df, dip_min, dip_max


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("solvent_mixture_features.py v4.0")
    print("=" * 60)
    print("\n用法:")
    print("  from solvent_mixture_features import add_mixture_features, add_salt_gates")
    print("\n电导率模型:")
    print("  df = add_mixture_features(df, task='conductivity')")
    print("\n粘度模型:")
    print("  df = add_mixture_features(df, task='viscosity')")
    print("  df, dip_min, dip_max = add_salt_gates(df)")
    print("\n查看溶剂表:")
    print("  get_solvent_table()")
