# TimesNet 太阳能功率预测项目

基于 TimesNet 的时序预测模型改进项目，用于太阳能功率预测任务。

## 项目结构

```
Time-Series-Library-timesnet/
├── models/                    # 模型定义
│   ├── TimesNet.py           # 原始TimesNet模型
│   ├── TimesNetPlus.py       # 增强版TimesNet+模型 (新增)
│   └── __init__.py
├── layers/                    # 模型组件层
│   ├── Conv_Blocks.py        # Inception卷积块
│   └── Embed.py              # 数据嵌入层
├── data_provider/             # 数据加载
│   ├── data_loader.py        # 数据集类 (已修改支持太阳能数据)
│   ├── data_factory.py       # 数据工厂 (已修改)
│   ├── DKA Solar Center dataset.csv  # 太阳能数据集1
│   └── site1.xlsx            # 太阳能数据集2
├── exp/                       # 训练实验
│   ├── exp_basic.py          # 基础实验类
│   └── exp_long_term_forecasting.py  # 长期预测实验
├── utils/                     # 工具函数
│   ├── metrics.py            # 评估指标
│   ├── tools.py              # 工具函数
│   ├── timefeatures.py       # 时间特征
│   ├── losses.py             # 损失函数
│   └── print_args.py         # 参数打印
├── run.py                    # 主入口 (已简化)
├── train_solar.py            # 太阳能预测训练脚本 (新增)
└── README.md                 # 本文档
```

## 数据集

项目支持两个太阳能数据集：
- **DKA Solar Center dataset.csv**: DKA太阳能中心数据集 (CSV格式)
  - 包含: Active_Power, Wind_Speed, Temperature, Humidity, Radiation等
  - 时间间隔: 5分钟
- **site1.xlsx**: 另一个太阳能站点数据 (Excel格式)

## 使用方法

### 方法1: 使用训练脚本

```bash
# 使用TimesNet模型训练
python train_solar.py --model TimesNet --data solar_dka

# 使用TimesNet+模型训练
python train_solar.py --model TimesNetPlus --data solar_dka

# 使用site1数据集
python train_solar.py --model TimesNetPlus --data solar_site1
```

### 方法2: 直接使用run.py

```bash
# TimesNet训练
python run.py --is_training 1 --model TimesNet --data solar_dka \
    --data_path "DKA Solar Center dataset.csv" \
    --target Active_Power --seq_len 96 --pred_len 96

# TimesNet+训练
python run.py --is_training 1 --model TimesNetPlus --data solar_dka \
    --data_path "DKA Solar Center dataset.csv" \
    --target Active_Power --seq_len 96 --pred_len 96

# 仅测试
python run.py --is_training 0 --model TimesNetPlus --data solar_dka \
    --data_path "DKA Solar Center dataset.csv"
```

## 模型改进方案

### TimesNet+ 核心改进

基于原始TimesNet，提出以下改进方案：

#### 1. 多尺度周期感知增强
- **改进**: 在FFT周期检测基础上，添加更多尺度的周期分析
- **效果**: 更好地捕捉太阳能数据的日内、日、周等不同时间尺度的周期性

#### 2. 时间序列分解
- **模块**: `SeriesDecomposition`
- **改进**: 添加趋势-季节性分解
- **效果**: 分离长期趋势和短期波动，增强预测稳定性

#### 3. 轻量级注意力机制
- **模块**: `LightweightAttention`
- **改进**: 在周期聚合阶段添加多头注意力
- **效果**: 自适应地融合不同周期的信息

#### 4. 自适应门控
- **模块**: `AdaptiveGating`
- **改进**: 动态调整特征权重
- **效果**: 自动筛选重要特征，抑制噪声

#### 5. 增强的Inception卷积
- **改进**: 使用更多尺度的卷积核
- **效果**: 捕捉更丰富的局部时序模式

## 论文创新点建议

### 潜在创新方向

1. **多尺度周期感知与注意力融合**
   - 创新点: 提出基于FFT的多周期检测结合轻量级注意力的融合机制
   - 贡献: 在保持计算效率的同时提升多周期建模能力

2. **时序分解增强的预测框架**
   - 创新点: 将趋势-季节性分解与深度周期感知相结合
   - 贡献: 解决太阳能等具有强周期性和趋势性数据的预测难题

3. **自适应特征选择机制**
   - 创新点: 门控机制自动学习特征重要性
   - 贡献: 减少人工特征工程，提升模型泛化能力

4. **针对太阳能预测的专门优化**
   - 创新点: 针对太阳能功率的特殊性（昼夜周期、天气影响）设计专门的周期检测机制
   - 贡献: 在太阳能预测任务上取得SOTA性能

## 实验配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| seq_len | 96 | 输入序列长度 (8小时 @ 5min) |
| pred_len | 96 | 预测序列长度 |
| d_model | 128 | 隐藏层维度 |
| e_layers | 2 | 模型层数 |
| batch_size | 32 | 批大小 |
| learning_rate | 0.001 | 学习率 |
| train_epochs | 20 | 训练轮数 |

## 评估指标

- MSE (均方误差)
- MAE (平均绝对误差)
- RMSE (均方根误差)

## 注意事项

1. 确保安装了必要的依赖: PyTorch, pandas, numpy, scikit-learn
2. 数据文件应放在 `data_provider/` 目录下
3. 训练结果保存在 `checkpoints/` 目录
4. 测试结果保存在 `results/` 目录
