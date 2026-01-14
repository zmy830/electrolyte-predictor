# -*- coding: utf-8 -*-
"""
solvent_database.py - 溶剂物性数据库管理

提供溶剂物性的查询、添加、修改功能
"""

import json
import os
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 内置溶剂数据库
# ============================================================================

BUILTIN_SOLVENTS = {
    # --- 环状碳酸酯 (Cyclic) ---
    "EC": {
        "name_cn": "碳酸乙烯酯",
        "type": "cyclic",
        "eps_r": 89.78,
        "eta_25": 1.90,
        "DN": 16.4,
        "AN": 18.0,
        "MW": 88.06,
        "rho": 1.32,
        "dipole": 5.35,
    },
    "PC": {
        "name_cn": "碳酸丙烯酯",
        "type": "cyclic",
        "eps_r": 64.92,
        "eta_25": 2.53,
        "DN": 15.1,
        "AN": 18.3,
        "MW": 102.09,
        "rho": 1.20,
        "dipole": 4.90,
    },
    "FEC": {
        "name_cn": "氟代碳酸乙烯酯",
        "type": "cyclic",
        "eps_r": 102.0,
        "eta_25": 4.10,
        "DN": 14.0,
        "AN": 20.0,
        "MW": 106.05,
        "rho": 1.45,
        "dipole": 5.60,
    },
    "VC": {
        "name_cn": "碳酸亚乙烯酯",
        "type": "cyclic",
        "eps_r": 126.0,
        "eta_25": 3.50,
        "DN": 15.0,
        "AN": 19.0,
        "MW": 86.05,
        "rho": 1.36,
        "dipole": 5.20,
    },
    "GBL": {
        "name_cn": "γ-丁内酯",
        "type": "cyclic",
        "eps_r": 39.0,
        "eta_25": 1.70,
        "DN": 18.0,
        "AN": 18.0,
        "MW": 86.09,
        "rho": 1.12,
        "dipole": 4.12,
    },
    
    # --- 链状碳酸酯 (Linear) ---
    "DMC": {
        "name_cn": "碳酸二甲酯",
        "type": "linear",
        "eps_r": 3.10,
        "eta_25": 0.59,
        "DN": 15.1,
        "AN": 14.0,
        "MW": 90.08,
        "rho": 1.07,
        "dipole": 0.90,
    },
    "EMC": {
        "name_cn": "碳酸甲乙酯",
        "type": "linear",
        "eps_r": 2.96,
        "eta_25": 0.65,
        "DN": 16.0,
        "AN": 14.0,
        "MW": 104.10,
        "rho": 1.01,
        "dipole": 0.89,
    },
    "DEC": {
        "name_cn": "碳酸二乙酯",
        "type": "linear",
        "eps_r": 2.80,
        "eta_25": 0.75,
        "DN": 16.0,
        "AN": 14.0,
        "MW": 118.13,
        "rho": 0.97,
        "dipole": 0.90,
    },
    
    # --- 羧酸酯 (Linear) ---
    "EA": {
        "name_cn": "乙酸乙酯",
        "type": "linear",
        "eps_r": 6.02,
        "eta_25": 0.43,
        "DN": 17.1,
        "AN": 14.0,
        "MW": 88.11,
        "rho": 0.90,
        "dipole": 1.78,
    },
    "MA": {
        "name_cn": "乙酸甲酯",
        "type": "linear",
        "eps_r": 6.68,
        "eta_25": 0.36,
        "DN": 16.5,
        "AN": 14.0,
        "MW": 74.08,
        "rho": 0.93,
        "dipole": 1.72,
    },
    "EP": {
        "name_cn": "丙酸乙酯",
        "type": "linear",
        "eps_r": 5.60,
        "eta_25": 0.52,
        "DN": 17.0,
        "AN": 12.0,
        "MW": 102.13,
        "rho": 0.89,
        "dipole": 1.74,
    },
    "PP": {
        "name_cn": "丙酸丙酯",
        "type": "linear",
        "eps_r": 4.70,
        "eta_25": 0.65,
        "DN": 17.0,
        "AN": 12.0,
        "MW": 116.16,
        "rho": 0.88,
        "dipole": 1.78,
    },
    "MF": {
        "name_cn": "甲酸甲酯",
        "type": "linear",
        "eps_r": 8.50,
        "eta_25": 0.33,
        "DN": 17.0,
        "AN": 15.0,
        "MW": 60.05,
        "rho": 0.97,
        "dipole": 1.77,
    },
    
    # --- 醚类 (Ether) ---
    "DME": {
        "name_cn": "乙二醇二甲醚",
        "type": "ether",
        "eps_r": 7.20,
        "eta_25": 0.46,
        "DN": 20.0,
        "AN": 10.0,
        "MW": 90.12,
        "rho": 0.87,
        "dipole": 1.71,
    },
    "DOL": {
        "name_cn": "1,3-二氧戊环",
        "type": "ether",
        "eps_r": 7.10,
        "eta_25": 0.58,
        "DN": 18.0,
        "AN": 10.0,
        "MW": 74.08,
        "rho": 1.06,
        "dipole": 1.90,
    },
    "THF": {
        "name_cn": "四氢呋喃",
        "type": "ether",
        "eps_r": 7.58,
        "eta_25": 0.46,
        "DN": 20.0,
        "AN": 8.0,
        "MW": 72.11,
        "rho": 0.89,
        "dipole": 1.75,
    },
    "Diglyme": {
        "name_cn": "二甘醇二甲醚",
        "type": "ether",
        "eps_r": 7.23,
        "eta_25": 0.98,
        "DN": 19.5,
        "AN": 10.0,
        "MW": 134.17,
        "rho": 0.94,
        "dipole": 1.97,
    },
    
    # --- 极性非质子 (Other) ---
    "AN": {
        "name_cn": "乙腈",
        "type": "other",
        "eps_r": 37.5,
        "eta_25": 0.34,
        "DN": 14.1,
        "AN": 18.9,
        "MW": 41.05,
        "rho": 0.78,
        "dipole": 3.92,
    },
    "DMF": {
        "name_cn": "N,N-二甲基甲酰胺",
        "type": "other",
        "eps_r": 36.7,
        "eta_25": 0.80,
        "DN": 26.6,
        "AN": 16.0,
        "MW": 73.09,
        "rho": 0.94,
        "dipole": 3.82,
    },
    "DMSO": {
        "name_cn": "二甲基亚砜",
        "type": "other",
        "eps_r": 46.7,
        "eta_25": 1.99,
        "DN": 29.8,
        "AN": 19.3,
        "MW": 78.13,
        "rho": 1.10,
        "dipole": 3.96,
    },
    "Sulfolane": {
        "name_cn": "环丁砜",
        "type": "other",
        "eps_r": 43.3,
        "eta_25": 10.3,
        "DN": 14.8,
        "AN": 19.0,
        "MW": 120.17,
        "rho": 1.26,
        "dipole": 4.35,
    },
}


# ============================================================================
# 盐数据库
# ============================================================================

SALT_DATABASE = {
    "LiPF6": {
        "name_cn": "六氟磷酸锂",
        "MW": 151.91,
        "conductivity_factor": 1.00,
        "description": "最常用的锂电池电解质盐，电导率高",
    },
    "LiFSI": {
        "name_cn": "双氟磺酰亚胺锂",
        "MW": 187.07,
        "conductivity_factor": 1.15,
        "description": "高电导率，低粘度增益，热稳定性好",
    },
    "LiTFSI": {
        "name_cn": "双三氟甲基磺酰亚胺锂",
        "MW": 287.09,
        "conductivity_factor": 0.95,
        "description": "高解离度，但阴离子大，粘度增加明显",
    },
    "LiBF4": {
        "name_cn": "四氟硼酸锂",
        "MW": 93.75,
        "conductivity_factor": 0.80,
        "description": "离子缔合严重，电导率较低",
    },
    "LiBOB": {
        "name_cn": "二草酸硼酸锂",
        "MW": 193.79,
        "conductivity_factor": 0.65,
        "description": "阴离子大，迁移慢，常作添加剂",
    },
    "LiDFOB": {
        "name_cn": "二氟草酸硼酸锂",
        "MW": 143.77,
        "conductivity_factor": 0.85,
        "description": "LiBOB 的氟化版本，性能介于 LiBF4 和 LiBOB",
    },
    "LiClO4": {
        "name_cn": "高氯酸锂",
        "MW": 106.39,
        "conductivity_factor": 0.95,
        "description": "解离度高，但有安全隐患",
    },
    "NONE": {
        "name_cn": "无盐",
        "MW": 0,
        "conductivity_factor": 0.01,
        "description": "纯溶剂体系",
    },
}


# ============================================================================
# 数据库管理类
# ============================================================================

class SolventDatabase:
    """溶剂物性数据库管理器"""
    
    def __init__(self, custom_json_path: Optional[str] = None):
        """
        初始化数据库
        
        Parameters
        ----------
        custom_json_path : str, optional
            自定义溶剂 JSON 文件路径
        """
        self.solvents = BUILTIN_SOLVENTS.copy()
        self.custom_path = custom_json_path
        
        if custom_json_path and os.path.exists(custom_json_path):
            self._load_custom_solvents(custom_json_path)
    
    def _load_custom_solvents(self, path: str):
        """加载自定义溶剂"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                custom = json.load(f)
            self.solvents.update(custom)
            print(f"Loaded {len(custom)} custom solvents from {path}")
        except Exception as e:
            print(f"Failed to load custom solvents: {e}")
    
    def get_solvent_list(self, category: Optional[str] = None) -> List[str]:
        """
        获取溶剂列表
        
        Parameters
        ----------
        category : str, optional
            按类型筛选: "cyclic", "linear", "ether", "other"
        
        Returns
        -------
        list
            溶剂名称列表
        """
        if category is None:
            return list(self.solvents.keys())
        
        return [
            name for name, props in self.solvents.items()
            if props.get("type") == category
        ]
    
    def get_properties(self, solvent_name: str) -> Optional[Dict]:
        """获取溶剂物性"""
        return self.solvents.get(solvent_name)
    
    def get_mw(self, solvent_name: str) -> Optional[float]:
        """获取分子量"""
        props = self.get_properties(solvent_name)
        return props.get("MW") if props else None
    
    def add_solvent(self, name: str, props: Dict) -> Tuple[bool, str]:
        """
        添加新溶剂
        
        Parameters
        ----------
        name : str
            溶剂名称
        props : dict
            物性数据
        
        Returns
        -------
        tuple
            (success: bool, message: str)
        """
        if name in self.solvents:
            return False, f"溶剂 {name} 已存在"
        
        # 验证物性
        is_valid, msg = self.validate_properties(props)
        if not is_valid:
            return False, msg
        
        self.solvents[name] = props
        self._save_custom_solvents()
        return True, f"成功添加溶剂 {name}"
    
    def update_solvent(self, name: str, props: Dict) -> Tuple[bool, str]:
        """更新溶剂数据"""
        if name not in self.solvents:
            return False, f"溶剂 {name} 不存在"
        
        is_valid, msg = self.validate_properties(props)
        if not is_valid:
            return False, msg
        
        self.solvents[name].update(props)
        self._save_custom_solvents()
        return True, f"成功更新溶剂 {name}"
    
    def delete_solvent(self, name: str) -> Tuple[bool, str]:
        """删除溶剂"""
        if name not in self.solvents:
            return False, f"溶剂 {name} 不存在"
        
        if name in BUILTIN_SOLVENTS:
            return False, f"内置溶剂 {name} 不能删除"
        
        del self.solvents[name]
        self._save_custom_solvents()
        return True, f"成功删除溶剂 {name}"
    
    def _save_custom_solvents(self):
        """保存自定义溶剂"""
        if not self.custom_path:
            return
        
        # 只保存非内置的溶剂
        custom = {
            k: v for k, v in self.solvents.items()
            if k not in BUILTIN_SOLVENTS
        }
        
        try:
            with open(self.custom_path, 'w', encoding='utf-8') as f:
                json.dump(custom, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save custom solvents: {e}")
    
    def validate_properties(self, props: Dict) -> Tuple[bool, str]:
        """验证物性数据"""
        required = ["eps_r", "eta_25", "MW", "type"]
        
        for field in required:
            if field not in props:
                return False, f"缺少必要字段: {field}"
        
        # 范围检查
        if not (1 < props.get("eps_r", 0) < 200):
            return False, f"介电常数超出范围: {props.get('eps_r')}"
        
        if not (0.1 < props.get("eta_25", 0) < 50):
            return False, f"粘度超出范围: {props.get('eta_25')}"
        
        if not (30 < props.get("MW", 0) < 500):
            return False, f"分子量超出范围: {props.get('MW')}"
        
        valid_types = ["cyclic", "linear", "ether", "other"]
        if props.get("type") not in valid_types:
            return False, f"无效溶剂类型: {props.get('type')}"
        
        return True, ""
    
    def export_to_json(self, path: str):
        """导出数据库"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.solvents, f, ensure_ascii=False, indent=2)
    
    def to_dataframe(self):
        """转换为 DataFrame"""
        import pandas as pd
        
        records = []
        for name, props in self.solvents.items():
            record = {"name": name}
            record.update(props)
            records.append(record)
        
        return pd.DataFrame(records)


# ============================================================================
# 盐管理
# ============================================================================

def get_salt_list() -> List[str]:
    """获取盐列表"""
    return list(SALT_DATABASE.keys())


def get_salt_info(salt_name: str) -> Optional[Dict]:
    """获取盐信息"""
    return SALT_DATABASE.get(salt_name)


def get_salt_conductivity_factor(salt_name: str) -> float:
    """获取盐的电导率修正系数"""
    info = SALT_DATABASE.get(salt_name)
    return info.get("conductivity_factor", 1.0) if info else 1.0
