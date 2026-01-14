# 模型文件目录

请将训练好的模型文件放置于此目录：

- `conductivity_catboost.pkl` - 电导率预测模型
- `viscosity_catboost.pkl` - 粘度预测模型

## 模型文件格式

模型文件应为 joblib 序列化的字典，包含：

```python
{
    "model": CatBoostRegressor,  # 训练好的模型
    "features": list,            # 特征列表
    "cat_cols": list,            # 类别特征列表
    "best_params": dict,         # 最佳超参数
    "metrics": dict,             # 评估指标
    
    # 粘度模型额外包含:
    "dip_min": float,            # dipole 归一化最小值
    "dip_max": float,            # dipole 归一化最大值
}
```

## 训练脚本

使用配套的训练脚本生成模型：

- `train_conductivity_whitelist.py` - 电导率模型训练
- `train_viscosity.py` - 粘度模型训练
