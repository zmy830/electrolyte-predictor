# 🔋 电解液配方性能预测系统

基于 CatBoost 机器学习模型的锂电池电解液配方性能预测系统。

## 功能特点

- ⚡ **电导率预测**: 盐伪装 + 物理修正策略
- 🌊 **粘度预测**: 残差模式 + Arrhenius 基线
- 🧪 **交互式配方设计**: 支持质量比/摩尔分数输入
- 📊 **批量预测**: CSV 文件批量处理
- 📚 **溶剂数据库**: 42 种溶剂物性数据

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置模型文件

将训练好的模型放入 `models/` 目录：

```
models/
├── conductivity_catboost.pkl   # 电导率模型
└── viscosity_catboost.pkl      # 粘度模型
```

### 3. 启动应用

```bash
streamlit run app.py
```

然后在浏览器中打开 http://localhost:8501

## 项目结构

```
electrolyte-predictor/
├── app.py                      # 主入口
├── requirements.txt            # 依赖
├── README.md                   # 说明文档
│
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── predictor.py            # 预测引擎
│   ├── formula_utils.py        # 配方工具
│   ├── solvent_database.py     # 溶剂数据库
│   └── solvent_mixture_features.py  # 特征工程
│
├── pages/                      # Streamlit 页面
│   ├── 1_📊_配方设计.py
│   ├── 2_🔬_特征原理.py
│   ├── 3_📚_溶剂库.py
│   └── 4_ℹ️_关于.py
│
├── models/                     # 模型文件 (需自行添加)
│   ├── conductivity_catboost.pkl
│   └── viscosity_catboost.pkl
│
└── data/                       # 数据文件
    └── custom_solvents.json    # 自定义溶剂 (可选)
```

## 使用指南

### 配方设计

1. 在侧边栏设置实验条件：
   - 温度 (支持 °C 或 K)
   - 盐种类 (LiPF6, LiFSI, LiTFSI 等)
   - 浓度和单位 (mol/L 或 mol/kg)

2. 在主面板选择溶剂：
   - 可选择预设配方或自定义
   - 支持质量比或摩尔分数输入
   - 自动归一化

3. 点击「开始预测」查看结果

### 批量预测

上传 CSV 文件，需包含以下列：
- `T_K`: 温度 (K)
- `salt_clean`: 盐种类
- `c_value`: 浓度
- `c_unit`: 浓度单位
- `frac_EC`, `frac_DMC`, ...: 溶剂摩尔分数

## 支持的盐种类

| 盐 | 电导率修正系数 | 说明 |
|----|---------------|------|
| LiPF6 | 1.00 | 基准 |
| LiFSI | 1.15 | 高电导率 |
| LiTFSI | 0.95 | 高解离度 |
| LiBF4 | 0.80 | 低电导率 |
| LiBOB | 0.65 | 阴离子大 |
| LiDFOB | 0.85 | 介于 BF4 和 BOB |
| LiClO4 | 0.95 | 解离度高 |

## 温度范围

- 最低: -40°C (233.15 K)
- 最高: 80°C (353.15 K)
- 默认: 25°C (298.15 K)

## 模型训练

如需重新训练模型，请使用配套的训练脚本：

- `train_conductivity.py`: 电导率模型训练
- `train_viscosity.py`: 粘度模型训练

## 技术栈

- **Web 框架**: Streamlit
- **机器学习**: CatBoost
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly, Matplotlib

## 许可证

MIT License
