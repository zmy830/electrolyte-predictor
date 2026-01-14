# -*- coding: utf-8 -*-
"""
formula_utils.py - 配方处理工具

提供配方转换、验证等功能
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


# ============================================================================
# 溶剂分子量表 (用于质量比转摩尔分数)
# ============================================================================

SOLVENT_MW = {
    # 环状碳酸酯
    "EC": 88.06, "PC": 102.09, "FEC": 106.05, "VC": 86.05, "GBL": 86.09,
    # 链状碳酸酯
    "DMC": 90.08, "EMC": 104.10, "DEC": 118.13,
    # 羧酸酯
    "EA": 88.11, "MA": 74.08, "EP": 102.13, "PP": 116.16, "MF": 60.05,
    # 醚类
    "DME": 90.12, "DOL": 74.08, "THF": 72.11, "2-MeTHF": 86.13,
    "Diglyme": 134.17, "Triglyme": 178.23, "Tetraglyme": 222.28,
    # 其他
    "AN": 41.05, "DMF": 73.09, "DMSO": 78.13, "Sulfolane": 120.17,
}


def mass_to_mole_fraction(
    mass_dict: Dict[str, float],
    mw_table: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    质量比 → 摩尔分数
    
    Parameters
    ----------
    mass_dict : dict
        质量百分比，如 {"EC": 30, "DMC": 70}
    mw_table : dict, optional
        分子量表，默认使用内置表
    
    Returns
    -------
    dict
        摩尔分数，如 {"EC": 0.28, "DMC": 0.72}
    
    Examples
    --------
    >>> mass_to_mole_fraction({"EC": 30, "DMC": 70})
    {'EC': 0.277, 'DMC': 0.723}
    """
    if mw_table is None:
        mw_table = SOLVENT_MW
    
    # 计算摩尔数 (质量 / 分子量)
    moles = {}
    for solvent, mass in mass_dict.items():
        if solvent not in mw_table:
            raise ValueError(f"未知溶剂: {solvent}，无法获取分子量")
        moles[solvent] = mass / mw_table[solvent]
    
    # 归一化为摩尔分数
    total_moles = sum(moles.values())
    if total_moles <= 0:
        raise ValueError("总摩尔数为零")
    
    fractions = {k: v / total_moles for k, v in moles.items()}
    return fractions


def mole_to_mass_fraction(
    mole_dict: Dict[str, float],
    mw_table: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    摩尔分数 → 质量比
    
    Parameters
    ----------
    mole_dict : dict
        摩尔分数
    mw_table : dict, optional
        分子量表
    
    Returns
    -------
    dict
        质量百分比
    """
    if mw_table is None:
        mw_table = SOLVENT_MW
    
    # 计算质量 (摩尔数 × 分子量)
    masses = {}
    for solvent, mole in mole_dict.items():
        if solvent not in mw_table:
            raise ValueError(f"未知溶剂: {solvent}，无法获取分子量")
        masses[solvent] = mole * mw_table[solvent]
    
    # 归一化为质量百分比
    total_mass = sum(masses.values())
    if total_mass <= 0:
        raise ValueError("总质量为零")
    
    percentages = {k: (v / total_mass) * 100 for k, v in masses.items()}
    return percentages


def normalize_fractions(frac_dict: Dict[str, float]) -> Dict[str, float]:
    """
    归一化分数使总和为1
    
    Parameters
    ----------
    frac_dict : dict
        分数字典
    
    Returns
    -------
    dict
        归一化后的分数
    """
    total = sum(frac_dict.values())
    if total <= 0:
        raise ValueError("分数总和为零")
    return {k: v / total for k, v in frac_dict.items()}


def validate_formula(
    formula: Dict[str, float],
    known_solvents: List[str] = None,
) -> Tuple[bool, str]:
    """
    验证配方有效性
    
    Parameters
    ----------
    formula : dict
        配方字典 (溶剂名: 分数)
    known_solvents : list, optional
        已知溶剂列表
    
    Returns
    -------
    tuple
        (is_valid: bool, error_msg: str)
    """
    if not formula:
        return False, "配方为空"
    
    # 检查分数值
    for solvent, frac in formula.items():
        if not isinstance(frac, (int, float)):
            return False, f"分数值无效: {solvent}={frac}"
        if frac < 0:
            return False, f"分数不能为负: {solvent}={frac}"
    
    # 检查总和
    total = sum(formula.values())
    if total <= 0:
        return False, "分数总和必须大于零"
    
    # 检查已知溶剂
    if known_solvents:
        unknown = [s for s in formula.keys() if s not in known_solvents]
        if unknown:
            return False, f"未知溶剂: {unknown}"
    
    return True, ""


def validate_conditions(
    T_K: float,
    conc: float,
    salt: str,
    conc_unit: str,
) -> Tuple[bool, str]:
    """
    验证实验条件
    
    Parameters
    ----------
    T_K : float
        温度 (K)
    conc : float
        浓度
    salt : str
        盐种类
    conc_unit : str
        浓度单位
    
    Returns
    -------
    tuple
        (is_valid: bool, error_msg: str)
    """
    # 温度范围检查
    if T_K < 233.15 or T_K > 353.15:  # -40°C to 80°C
        return False, f"温度超出范围: {T_K} K ({T_K-273.15:.1f}°C)，建议范围 -40°C ~ 80°C"
    
    # 浓度检查
    if conc < 0:
        return False, f"浓度不能为负: {conc}"
    if conc > 5:
        return False, f"浓度过高: {conc}，建议不超过 5 mol/L"
    
    # 盐种类检查
    known_salts = ["LiPF6", "LiFSI", "LiTFSI", "LiBF4", "LiBOB", "LiDFOB", "LiTDI", "LiClO4", "NONE"]
    if salt not in known_salts:
        return False, f"未知盐种类: {salt}"
    
    # 浓度单位检查
    valid_units = ["mol/L", "mol/kg", "M", "m"]
    if conc_unit not in valid_units:
        return False, f"无效浓度单位: {conc_unit}，支持: {valid_units}"
    
    return True, ""


def format_formula_string(formula: Dict[str, float], style: str = "ratio") -> str:
    """
    格式化配方为字符串
    
    Parameters
    ----------
    formula : dict
        配方字典
    style : str
        格式化风格: "ratio" (比例), "percent" (百分比)
    
    Returns
    -------
    str
        格式化字符串
    
    Examples
    --------
    >>> format_formula_string({"EC": 0.3, "DMC": 0.7}, "ratio")
    "EC:DMC = 3:7"
    """
    if not formula:
        return ""
    
    # 按分数排序
    sorted_items = sorted(formula.items(), key=lambda x: -x[1])
    
    if style == "ratio":
        # 转换为最简整数比
        fracs = [f for _, f in sorted_items]
        # 找最小值归一化
        min_frac = min(f for f in fracs if f > 0)
        ratios = [round(f / min_frac, 1) for f in fracs]
        
        # 尝试转整数
        scale = 1
        for r in ratios:
            if r != int(r):
                scale = 10
                break
        ratios = [int(r * scale) for r in ratios]
        
        # 约分
        from math import gcd
        from functools import reduce
        common = reduce(gcd, ratios)
        ratios = [r // common for r in ratios]
        
        names = [name for name, _ in sorted_items]
        return ":".join(names) + " = " + ":".join(map(str, ratios))
    
    elif style == "percent":
        parts = [f"{name} {frac*100:.1f}%" for name, frac in sorted_items]
        return ", ".join(parts)
    
    else:
        return str(formula)


# ============================================================================
# 预设配方
# ============================================================================

PRESET_FORMULAS = {
    "EC:DMC 3:7": {"EC": 0.3, "DMC": 0.7},
    "EC:EMC 3:7": {"EC": 0.3, "EMC": 0.7},
    "EC:DEC 3:7": {"EC": 0.3, "DEC": 0.7},
    "EC:DMC:EMC 1:1:1": {"EC": 1/3, "DMC": 1/3, "EMC": 1/3},
    "EC:DMC:DEC 1:1:1": {"EC": 1/3, "DMC": 1/3, "DEC": 1/3},
    "PC:DMC 1:1": {"PC": 0.5, "DMC": 0.5},
    "EC:EP 3:7": {"EC": 0.3, "EP": 0.7},
    "EC:EA 3:7": {"EC": 0.3, "EA": 0.7},
}


def get_preset_formula(name: str) -> Optional[Dict[str, float]]:
    """获取预设配方"""
    return PRESET_FORMULAS.get(name)


def list_preset_formulas() -> List[str]:
    """列出所有预设配方"""
    return list(PRESET_FORMULAS.keys())
