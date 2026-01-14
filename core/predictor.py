# -*- coding: utf-8 -*-
"""
predictor.py - 电解液性能预测引擎

封装电导率和粘度预测的完整流程
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from .solvent_mixture_features import add_mixture_features, add_salt_gates


# ============================================================================
# 盐修正因子配置 (电导率用)
# ============================================================================

SALT_CORRECTION_FACTORS = {
    "LiPF6": 1.00,
    "LiFSI": 1.15,
    "LiTFSI": 0.95,
    "LiBF4": 0.80,
    "LiBOB": 0.65,
    "LiDFOB": 0.85,
    "LiTDI": 0.70,
    "LiClO4": 0.95,
    "NONE": 0.01,
}


def get_salt_correction_factor(salt_name: str) -> float:
    """获取盐的电导率修正系数"""
    return SALT_CORRECTION_FACTORS.get(salt_name, 1.0)


# ============================================================================
# 预测器类
# ============================================================================

class ElectrolytePredictor:
    """电解液性能预测器"""
    
    def __init__(
        self,
        conductivity_model_path: Optional[str] = None,
        viscosity_model_path: Optional[str] = None,
    ):
        """
        初始化预测器
        
        Parameters
        ----------
        conductivity_model_path : str
            电导率模型路径 (pkl 文件)
        viscosity_model_path : str
            粘度模型路径 (pkl 文件)
        """
        self.conductivity_model = None
        self.conductivity_features = None
        self.conductivity_cat_cols = []
        
        self.viscosity_model = None
        self.viscosity_features = None
        self.viscosity_cat_cols = []
        self.viscosity_dip_min = 0.0
        self.viscosity_dip_max = 5.0
        
        if conductivity_model_path and os.path.exists(conductivity_model_path):
            self._load_conductivity_model(conductivity_model_path)
        
        if viscosity_model_path and os.path.exists(viscosity_model_path):
            self._load_viscosity_model(viscosity_model_path)
    
    def _load_conductivity_model(self, path: str):
        """加载电导率模型"""
        bundle = joblib.load(path)
        if isinstance(bundle, dict):
            self.conductivity_model = bundle.get("model", bundle)
            self.conductivity_features = bundle.get("features", None)
            self.conductivity_cat_cols = bundle.get("cat_cols", [])
        else:
            self.conductivity_model = bundle
            self.conductivity_features = None
        print(f"[Conductivity] Model loaded from {path}")
    
    def _load_viscosity_model(self, path: str):
        """加载粘度模型"""
        bundle = joblib.load(path)
        if isinstance(bundle, dict):
            self.viscosity_model = bundle.get("model", bundle)
            self.viscosity_features = bundle.get("features", None)
            self.viscosity_cat_cols = bundle.get("cat_cols", [])
            self.viscosity_dip_min = bundle.get("dip_min", 0.0)
            self.viscosity_dip_max = bundle.get("dip_max", 5.0)
        else:
            self.viscosity_model = bundle
        print(f"[Viscosity] Model loaded from {path}")
    
    def _prepare_dataframe(
        self,
        formula: Dict[str, float],
        T_K: float,
        salt: str,
        conc: float,
        conc_unit: str = "mol/L",
    ) -> pd.DataFrame:
        """
        将配方转换为 DataFrame
        
        Parameters
        ----------
        formula : dict
            溶剂摩尔分数，如 {"EC": 0.3, "DMC": 0.7}
        T_K : float
            温度 (K)
        salt : str
            盐种类
        conc : float
            浓度值
        conc_unit : str
            浓度单位 ("mol/L" 或 "mol/kg")
        
        Returns
        -------
        DataFrame
        """
        # 构建基础数据
        data = {
            "T_K": [T_K],
            "c_value": [conc],
            "c_units": [conc_unit],
            "c_unit": [conc_unit],  # 兼容两种命名
            "salt_clean": [salt],
        }
        
        # 添加溶剂分数
        for solvent, frac in formula.items():
            data[f"frac_{solvent}"] = [frac]
        
        df = pd.DataFrame(data)
        return df
    
    def predict_conductivity(
        self,
        formula: Dict[str, float],
        T_K: float = 298.15,
        salt: str = "LiPF6",
        conc: float = 1.0,
        conc_unit: str = "mol/L",
    ) -> Dict:
        """
        预测电导率
        
        Parameters
        ----------
        formula : dict
            溶剂摩尔分数
        T_K : float
            温度 (K)，默认 298.15 (25°C)
        salt : str
            盐种类
        conc : float
            浓度
        conc_unit : str
            浓度单位
        
        Returns
        -------
        dict
            {
                'k_pred_final': float,      # 最终电导率 (mS/cm)
                'k_pred_base': float,       # LiPF6当量值
                'log_k_pred': float,        # ln(k)
                'salt_correction': float,   # 盐修正系数
                'success': bool,
                'error': str or None,
            }
        """
        if self.conductivity_model is None:
            return {
                'k_pred_final': None,
                'k_pred_base': None,
                'log_k_pred': None,
                'salt_correction': None,
                'success': False,
                'error': "电导率模型未加载",
            }
        
        try:
            # 准备数据
            df = self._prepare_dataframe(formula, T_K, salt, conc, conc_unit)
            
            # 计算混合物特征
            try:
                df = add_mixture_features(df, frac_prefix="frac_", T_col="T_K", task="conductivity")
            except TypeError:
                df = add_mixture_features(df, frac_prefix="frac_", T_col="T_K")
            
            # 去除重复列
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
            
            # 盐伪装策略
            df_masked = df.copy()
            df_masked["salt_clean"] = "LiPF6"
            
            # 准备特征
            if self.conductivity_features:
                # 补齐缺失特征
                for col in self.conductivity_features:
                    if col not in df_masked.columns:
                        df_masked[col] = 0.0
                X = df_masked[self.conductivity_features].copy()
            else:
                X = df_masked.copy()
            
            # 类别特征转换
            for c in self.conductivity_cat_cols:
                if c in X.columns:
                    X[c] = X[c].astype(str)
            
            # 预测
            log_k_pred = float(self.conductivity_model.predict(X)[0])
            k_base = float(np.exp(log_k_pred))
            
            # 盐修正
            salt_corr = get_salt_correction_factor(salt)
            k_final = k_base * salt_corr
            
            return {
                'k_pred_final': k_final,
                'k_pred_base': k_base,
                'log_k_pred': log_k_pred,
                'salt_correction': salt_corr,
                'success': True,
                'error': None,
            }
            
        except Exception as e:
            return {
                'k_pred_final': None,
                'k_pred_base': None,
                'log_k_pred': None,
                'salt_correction': None,
                'success': False,
                'error': str(e),
            }
    
    def predict_viscosity(
        self,
        formula: Dict[str, float],
        T_K: float = 298.15,
        salt: str = "LiPF6",
        conc: float = 1.0,
        conc_unit: str = "mol/L",
    ) -> Dict:
        """
        预测粘度
        
        Parameters
        ----------
        formula : dict
            溶剂摩尔分数
        T_K : float
            温度 (K)
        salt : str
            盐种类
        conc : float
            浓度
        conc_unit : str
            浓度单位
        
        Returns
        -------
        dict
            {
                'eta_pred': float,          # 粘度 (mPa·s)
                'ln_eta_pred': float,       # ln(η)
                'ln_eta_ideal': float,      # Arrhenius基线
                'residual': float,          # 残差修正
                'success': bool,
                'error': str or None,
            }
        """
        if self.viscosity_model is None:
            return {
                'eta_pred': None,
                'ln_eta_pred': None,
                'ln_eta_ideal': None,
                'residual': None,
                'success': False,
                'error': "粘度模型未加载",
            }
        
        try:
            # 准备数据
            df = self._prepare_dataframe(formula, T_K, salt, conc, conc_unit)
            
            # 计算混合物特征
            try:
                df = add_mixture_features(df, frac_prefix="frac_", T_col="T_K", task="viscosity")
            except TypeError:
                df = add_mixture_features(df, frac_prefix="frac_", T_col="T_K")
            
            # 去除重复列
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
            
            # 获取 Arrhenius 基线
            if "ln_eta_ideal_T" not in df.columns:
                # 如果旧版没有这个特征，用 ln_eta0_mix_25C 或 ln_eta0_mix 近似
                for col in ["ln_eta0_mix_25C", "ln_eta0_mix"]:
                    if col in df.columns:
                        df["ln_eta_ideal_T"] = df[col]
                        break
                else:
                    df["ln_eta_ideal_T"] = 0.0
            
            ln_eta_ideal = float(df["ln_eta_ideal_T"].iloc[0])
            
            # 添加盐效应门控特征
            df, _, _ = add_salt_gates(
                df,
                dip_min=self.viscosity_dip_min,
                dip_max=self.viscosity_dip_max,
            )
            
            # 准备特征
            if self.viscosity_features:
                for col in self.viscosity_features:
                    if col not in df.columns:
                        df[col] = 0.0
                X = df[self.viscosity_features].copy()
            else:
                X = df.copy()
            
            # 类别特征转换
            for c in self.viscosity_cat_cols:
                if c in X.columns:
                    X[c] = X[c].astype(str)
            
            # 预测残差
            residual = float(self.viscosity_model.predict(X)[0])
            
            # 重构 ln(η)
            ln_eta_pred = residual + ln_eta_ideal
            eta_pred = float(np.exp(ln_eta_pred))
            
            return {
                'eta_pred': eta_pred,
                'ln_eta_pred': ln_eta_pred,
                'ln_eta_ideal': ln_eta_ideal,
                'residual': residual,
                'success': True,
                'error': None,
            }
            
        except Exception as e:
            return {
                'eta_pred': None,
                'ln_eta_pred': None,
                'ln_eta_ideal': None,
                'residual': None,
                'success': False,
                'error': str(e),
            }
    
    def predict_both(
        self,
        formula: Dict[str, float],
        T_K: float = 298.15,
        salt: str = "LiPF6",
        conc: float = 1.0,
        conc_unit: str = "mol/L",
    ) -> Dict:
        """同时预测电导率和粘度"""
        cond_result = self.predict_conductivity(formula, T_K, salt, conc, conc_unit)
        visc_result = self.predict_viscosity(formula, T_K, salt, conc, conc_unit)
        
        return {
            'conductivity': cond_result,
            'viscosity': visc_result,
            'formula': formula,
            'conditions': {
                'T_K': T_K,
                'T_C': T_K - 273.15,
                'salt': salt,
                'concentration': conc,
                'unit': conc_unit,
            }
        }
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测
        
        Parameters
        ----------
        df : DataFrame
            需包含: T_K, salt_clean, c_value, c_unit, frac_* 列
        
        Returns
        -------
        DataFrame
            新增预测结果列
        """
        results = []
        
        for idx, row in df.iterrows():
            # 提取溶剂分数
            formula = {}
            for col in df.columns:
                if col.startswith("frac_") and col not in ["frac_cyclic", "frac_linear", "frac_ether"]:
                    val = row[col]
                    if pd.notna(val) and val > 0:
                        solvent = col[5:]  # 去掉 "frac_"
                        formula[solvent] = float(val)
            
            # 获取条件
            T_K = float(row.get("T_K", 298.15))
            salt = str(row.get("salt_clean", "LiPF6"))
            conc = float(row.get("c_value", 1.0))
            conc_unit = str(row.get("c_unit", row.get("c_units", "mol/L")))
            
            # 预测
            result = self.predict_both(formula, T_K, salt, conc, conc_unit)
            
            results.append({
                'k_pred': result['conductivity'].get('k_pred_final'),
                'k_pred_base': result['conductivity'].get('k_pred_base'),
                'salt_corr': result['conductivity'].get('salt_correction'),
                'eta_pred': result['viscosity'].get('eta_pred'),
                'ln_eta_ideal': result['viscosity'].get('ln_eta_ideal'),
            })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# ============================================================================
# 便捷函数
# ============================================================================

def create_predictor(
    conductivity_model_path: str = None,
    viscosity_model_path: str = None,
) -> ElectrolytePredictor:
    """创建预测器实例"""
    return ElectrolytePredictor(
        conductivity_model_path=conductivity_model_path,
        viscosity_model_path=viscosity_model_path,
    )
