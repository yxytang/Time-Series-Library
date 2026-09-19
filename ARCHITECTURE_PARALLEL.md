# TimesNet-Full 并行架构设计

## 🎯 核心设计理念

**并行双分支架构**：空间周期建模 + 时间依赖建模同时进行，然后自适应融合

---

## 📐 完整架构图

```
Input [B, seq_len, enc_in]
    ↓
┌─────────────────────┐
│  STL Decomposition  │  (period=24h)
│  Trend + Seasonal   │
│    + Residual       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Triple Embeddings   │  (独立embedding)
│   + Summation       │
└─────────────────────┘
    ↓
[B, seq_len+pred_len, d_model]
    ↓
┌─────────────────────────────────────────┐
│        Parallel Dual-Branch Block       │
│                                         │
│  ┌───────────────┐  ┌───────────────┐  │
│  │   Branch 1    │  │   Branch 2    │  │
│  │ DualDomain    │  │ AutoCorr      │  │
│  │ TimesBlock    │  │ (Temporal)    │  │
│  │ (Spatial 2D)  │  │               │  │
│  └───────┬───────┘  └───────┬───────┘  │
│          │                  │          │
│          └────────┬─────────┘          │
│                   ↓                    │
│         ┌────────────────┐             │
│         │ Adaptive Gate  │             │
│         │   Fusion       │             │
│         └────────┬───────┘             │
│                  ↓                     │
│         Residual Connection            │
│                  ↓                     │
│            LayerNorm                   │
└─────────────────┼───────────────────────┘
                  ↓
            (Repeat N layers)
                  ↓
┌─────────────────────┐
│    Projection       │
└─────────────────────┘
    ↓
Output [B, pred_len, c_out]
```

---

## 🔧 各模块详细说明

### 1️⃣ STL分解模块

```python
class SimpleSTL:
    period = 24  # 单周期（24小时）
    
    def forward(x):
        trend = moving_average(x, window=period)
        seasonal = extract_seasonal(x - trend, period)
        residual = x - trend - seasonal
        return trend, seasonal, residual
```

**作用**：分离趋势、周期和噪声

---

### 2️⃣ 并行分支1：DualDomainTimesBlock

```python
# 空间2D周期建模
FFT Periods ───→ 2D Reshape ───→ Inception Conv ─┐
                                                   ├→ Gate Fusion
DCT Periods ───→ 2D Reshape ───→ Inception Conv ─┘
```

**特点**：
- FFT检测周期（全局频谱）
- DCT检测周期（局部模式）
- 双路Inception卷积
- 自适应门控融合

---

### 3️⃣ 并行分支2：AutoCorrelation

```python
# 时间依赖建模
Q, K, V = Linear(x)
Correlation = FFT(Q) * conj(FFT(K))
Top-k delays = topk(Correlation)
Output = time_delay_aggregation(V, delays)
```

**特点**：
- 基于自相关的时间依赖
- 捕获跨周期关系
- 多头机制（8 heads）

---

### 4️⃣ 自适应门控融合

```python
class AdaptiveGateFusion:
    def forward(spatial_out, temporal_out):
        concat = cat([spatial_out, temporal_out], dim=-1)
        # [B, T, 2*d_model]
        
        gate = Softmax(
            Linear(Tanh(Linear(concat)))
        )  # [B, T, 2]
        
        output = gate[...,0:1] * spatial_out + 
                 gate[...,1:2] * temporal_out
        return output
```

**优势**：
- 动态权重分配
- 时间步级别的自适应
- 端到端学习

---

## 🆚 并行 vs 串行架构对比

| 维度 | 串行架构 | 并行架构 |
|------|----------|----------|
| **信息流** | TimesBlock → AutoCorr | TimesBlock ∥ AutoCorr → Fusion |
| **计算效率** | 顺序执行 | 可并行计算 |
| **特征融合** | 累加残差 | 自适应门控 |
| **梯度流** | 单路径 | 双路径（更稳定） |
| **表达能力** | 分层抽象 | 多视角融合 |

---

## 🎨 关键创新点

### ✅ 1. 双视角建模
- **空间视角**：2D CNN捕获周期内的空间模式
- **时间视角**：AutoCorr捕获跨周期的时间依赖

### ✅ 2. 自适应融合
- 不同数据集/时间段，两个分支的重要性不同
- 门控网络自动学习最优权重

### ✅ 3. 残差连接
- `enc_out = enc_out + fused_out`
- 保证梯度流畅，避免退化

### ✅ 4. 层次化设计
- 浅层：局部模式学习
- 深层：全局依赖整合

---

## 📊 预期性能提升

| 改进项 | 预期增益 | 原因 |
|--------|----------|------|
| **并行计算** | 训练加速1.3x | 两分支可GPU并行 |
| **特征融合** | MSE -2~3% | 门控优于简单相加 |
| **梯度稳定性** | 更快收敛 | 双路径缓解梯度消失 |
| **泛化能力** | 更鲁棒 | 多视角特征互补 |

---

## 🔬 消融实验设计

### Variant 1: 只用空间分支
```python
use_autocorr = False
# 只保留DualDomainTimesBlock
```

### Variant 2: 只用时间分支
```python
# 只保留AutoCorrelation
```

### Variant 3: 简单相加融合
```python
fused_out = 0.5 * spatial_out + 0.5 * temporal_out
```

### Variant 4: 完整并行（Ours）
```python
# 自适应门控融合
```

---

## 💡 实现细节

### 参数量分析
```
STL分解: 0 (无参数)
Embeddings: 3 × (enc_in × d_model) = 3 × (7 × 64) = 1,344
DualDomainTimesBlock × N: ~15K × 2 (层数)
AutoCorrelation × N: ~8K × 2 (层数)
Gate Fusion × N: 2 × d_model² × 2 = 2 × 4096 × 2 = 16,384
Projection: d_model × c_out = 64 × 7 = 448

Total: ~60K parameters (2层示例)
```

### 计算复杂度
- **空间分支**: O(T × N × C²) (CNN)
- **时间分支**: O(T log T × d_model) (FFT)
- **门控融合**: O(T × d_model²)
- **总复杂度**: O(T × max(N × C², d_model²))

---

## 🚀 训练策略

### 1. Warmup阶段（Epoch 1-5）
- 先冻结门控网络
- 让两个分支各自学习特征

### 2. Joint Training（Epoch 6+）
- 解冻所有参数
- 端到端优化

### 3. 学习率调度
```python
spatial_branch_lr = 1e-4
temporal_branch_lr = 1e-4
gate_fusion_lr = 1e-3  # 更快适应
```

---

## 📝 论文撰写要点

### Method部分
1. 用图示清晰展示并行架构
2. 强调"互补性"而非"竞争性"
3. 数学公式描述门控机制

### Experiment部分
1. 消融实验证明并行优于串行
2. 可视化门控权重分布
3. 不同数据集的权重差异分析

### Discussion部分
1. 分析什么情况下空间分支更重要
2. 分析什么情况下时间分支更重要
3. 讨论可扩展性（3分支、4分支...）

---

## ✅ 总结

**并行架构的核心优势**：
1. ✅ 多视角特征提取
2. ✅ 自适应权重学习
3. ✅ 梯度流更稳定
4. ✅ 可并行计算加速
5. ✅ 理论上更优雅（对称性）

**适用场景**：
- 周期性 + 趋势性同时显著的数据
- 需要多尺度建模的任务
- 要求高精度的长期预测
