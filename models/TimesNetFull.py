"""
TimesNet-Full: 完整论文方案
集成三大创新点：
1. STL分解（标准单周期24h）
2. FFT+DCT双域互补
3. AutoCorrelation跨周期注意力

论文标题:
"Dual-Domain TimesNet with STL Decomposition and Auto-Correlation
 for Long-Term Solar Forecasting"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


# ============================================================================
# 模块1: STL分解 (Seasonal-Trend-Loess)
# ============================================================================

class STLDecomposition(nn.Module):
    """
    STL-style seasonal-trend decomposition.

    impl='torch' (default): vectorized, fully differentiable, runs on GPU.
        trend    = centered moving average (replicate-padded)
        seasonal = per-phase average across cycles of the de-trended series
        residual = x - trend - seasonal
        Two-pass (de-seasonalize then re-estimate trend), mirroring STL's
        inner loop, so the components stay orthogonal.

    impl='statsmodels': the original LOESS-based STL. Accurate but must be
        applied per (sample, variable) on CPU -- ~B*C fits per forward, so it
        is only usable for small C or offline pre-computation.
    """

    def __init__(self, period=24, seasonal=13, impl='torch', trend_kernel=None):
        super(STLDecomposition, self).__init__()
        self.period = period
        self.seasonal = seasonal
        self.impl = impl
        if trend_kernel is None:
            # statsmodels' default trend window: next odd int >=
            # 1.5 * period / (1 - 1.5 / seasonal)
            trend_kernel = int(np.ceil(1.5 * period / (1.0 - 1.5 / seasonal)))
        if trend_kernel % 2 == 0:
            trend_kernel += 1
        self.trend_kernel = trend_kernel

    # ------------------------------------------------------------------
    # vectorized torch path
    # ------------------------------------------------------------------
    def _moving_average(self, x):
        """x: [B, T, C] -> trend [B, T, C]"""
        k = self.trend_kernel
        pad = k // 2
        xp = x.permute(0, 2, 1)                       # [B, C, T]
        xp = F.pad(xp, (pad, pad), mode='replicate')
        trend = F.avg_pool1d(xp, kernel_size=k, stride=1)
        return trend.permute(0, 2, 1)

    def _seasonal(self, x):
        """x: [B, T, C] (already de-trended) -> per-phase mean, tiled to T"""
        B, T, C = x.shape
        p = self.period
        n_cycles = T // p
        if n_cycles < 2:
            return torch.zeros_like(x)
        core = x[:, :n_cycles * p, :].reshape(B, n_cycles, p, C).mean(dim=1)
        reps = (T + p - 1) // p
        return core.repeat(1, reps, 1)[:, :T, :]

    def _forward_torch(self, x):
        detrended = x - self._moving_average(x)
        seasonal = self._seasonal(detrended)
        trend = self._moving_average(x - seasonal)
        residual = x - trend - seasonal
        return trend, seasonal, residual

    # ------------------------------------------------------------------
    # statsmodels path (slow, non-differentiable, kept for validation)
    # ------------------------------------------------------------------
    def _forward_statsmodels(self, x):
        from statsmodels.tsa.seasonal import STL

        B, T, C = x.shape
        device = x.device
        x_np = x.detach().cpu().numpy()
        trend = np.empty((B, T, C), dtype=np.float32)
        seasonal = np.empty((B, T, C), dtype=np.float32)
        residual = np.empty((B, T, C), dtype=np.float32)
        for c in range(C):
            for b in range(B):
                series = x_np[b, :, c]
                if T >= 2 * self.period:
                    res = STL(series, period=self.period,
                              seasonal=self.seasonal).fit()
                    trend[b, :, c] = res.trend
                    seasonal[b, :, c] = res.seasonal
                    residual[b, :, c] = res.resid
                else:
                    trend[b, :, c] = series
        to_t = lambda a: torch.from_numpy(a).to(device)
        return to_t(trend), to_t(seasonal), to_t(residual)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            trend, seasonal, residual: [B, T, C] each
        """
        if self.impl == 'statsmodels':
            return self._forward_statsmodels(x)
        return self._forward_torch(x)


# ============================================================================
# 模块2: FFT+DCT双域分析
# ============================================================================

_DCT_BASIS_CACHE = {}


def _dct_ii_basis(n, device, dtype):
    """Cached DCT-II cosine basis D[k, t] = cos(pi*k*(2t+1) / (2n))."""
    key = (n, str(device), str(dtype))
    basis = _DCT_BASIS_CACHE.get(key)
    if basis is None:
        t = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos(math.pi * k * (2 * t + 1) / (2 * n))
        _DCT_BASIS_CACHE[key] = basis
    return basis


def dct_ii(x):
    """
    DCT-II along the time dimension: x [B, T, C] -> [B, T, C].
    Matches the definition in the paper (Eq. 3.4): uses the half-sample
    offset, i.e. the even-symmetric (mirror) extension about t = -0.5.
    """
    T = x.shape[1]
    basis = _dct_ii_basis(T, x.device, x.dtype)          # [T, T]
    out = torch.matmul(x.transpose(1, 2), basis.transpose(0, 1))
    return out.transpose(1, 2)


def _top_indices(amp, k):
    """Indices of the k largest bins (numpy)."""
    return torch.topk(amp, k)[1].detach().cpu().numpy()


def dual_domain_period_detection(x, k=5, use_dct=True):
    """
    FFT + DCT dual-domain period detection.

    Bin -> period mapping differs between the two transforms and this matters:
      * FFT  : basis e^{-2*pi*i*k*t/T} has k full cycles over the window
               => period = T / k
      * DCT-II: basis cos(pi*k*(2t+1)/(2T)) has k/2 cycles over the window
               => period = 2T / k          (NOT T / k)
    Using T/k for the DCT path halves every detected period, which puts the two
    branches on different period scales and makes the folding misaligned.

    Args:
        x: [B, T, C]
        k: number of top periods per domain
        use_dct: if False, only the FFT domain is computed
    Returns:
        fft_periods: [k], fft_weights: [B, k]
        dct_periods: [k] or None, dct_weights: [B, k] or None
    """
    B, T, C = x.shape

    # --- FFT domain: complex-exponential basis, period = T / k ---
    fft_spec = torch.fft.rfft(x, dim=1).abs()          # [B, T//2+1, C]
    fft_amp = fft_spec.mean(0).mean(-1).clone()        # [T//2+1]
    fft_amp[0] = 0                                     # drop DC
    fft_idx = _top_indices(fft_amp, k)
    fft_periods = np.clip(T // fft_idx, 2, T)
    fft_weights = fft_spec.mean(-1)[:, torch.as_tensor(fft_idx, device=x.device)]

    if not use_dct:
        return fft_periods, fft_weights, None, None

    # --- DCT domain: real cosine basis, period = 2T / k ---
    dct_spec = dct_ii(x).abs()                         # [B, T, C]
    dct_amp = dct_spec.mean(0).mean(-1).clone()        # [T]
    dct_amp[0] = 0                                     # drop DC
    # k=1 corresponds to period 2T, which exceeds the window and cannot be
    # folded; start the search at k>=2 so every candidate period is <= T.
    dct_amp[1] = 0
    dct_idx = _top_indices(dct_amp, k)
    dct_periods = np.clip(2 * T // dct_idx, 2, T)
    dct_weights = dct_spec.mean(-1)[:, torch.as_tensor(dct_idx, device=x.device)]

    return fft_periods, fft_weights, dct_periods, dct_weights


# ============================================================================
# 模块3: 双域TimesBlock
# ============================================================================

class DualDomainTimesBlock(nn.Module):
    """
    FFT+DCT双域TimesBlock
    
    创新点:
    1. 并行FFT和DCT周期检测
    2. 双路2D CNN处理
    3. 自适应门控融合
    """
    def __init__(self, configs):
        super(DualDomainTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.use_dct = getattr(configs, 'use_dct', True)

        # FFT路径的CNN
        self.fft_conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                             num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                             num_kernels=configs.num_kernels)
        )

        # DCT路径的CNN（仅 use_dct 时创建，使"仅FFT vs FFT+DCT"消融可直接运行）
        if self.use_dct:
            self.dct_conv = nn.Sequential(
                Inception_Block_V1(configs.d_model, configs.d_ff,
                                 num_kernels=configs.num_kernels),
                nn.GELU(),
                Inception_Block_V1(configs.d_ff, configs.d_model,
                                 num_kernels=configs.num_kernels)
            )
            # 双域融合：两路输出拼接后经全连接层
            self.fuse = nn.Linear(configs.d_model * 2, configs.d_model)
        else:
            self.dct_conv = None
            self.fuse = None
    
    def process_domain(self, x, period_list, period_weight, conv_module):
        """处理单个域（FFT或DCT）"""
        B, T, N = x.size()
        res = []
        
        for i in range(self.k):
            period = period_list[i]
            
            # Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            # Reshape to 2D
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D Conv
            out = conv_module(out)
            
            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # 加权聚合
        res = torch.stack(res, dim=-1)  # [B, T, N, k]
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        return res
    
    def forward(self, x):
        """
        Args:
            x: [B, T, N]
        Returns:
            out: [B, T, N]
        """
        B, T, N = x.size()

        # 双域周期检测（两条通路各取各自的周期集合）
        fft_periods, fft_weights, dct_periods, dct_weights = \
            dual_domain_period_detection(x, self.k, use_dct=self.use_dct)

        # FFT路径处理
        fft_out = self.process_domain(x, fft_periods, fft_weights, self.fft_conv)

        if not self.use_dct:
            return fft_out + x

        # DCT路径处理
        dct_out = self.process_domain(x, dct_periods, dct_weights, self.dct_conv)

        # 双域融合：两路输出沿特征维拼接后经全连接层
        fused_out = self.fuse(torch.cat([fft_out, dct_out], dim=-1))  # [B, T, N]

        return fused_out + x


# ============================================================================
# 模块4: AutoCorrelation (来自Autoformer)
# ============================================================================

class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism (Autoformer, NeurIPS 2021).

    Vectorized, Autoformer-faithful: the top-k delays are selected once from
    the batch-mean correlation spectrum and applied with torch.roll, so the
    aggregation costs top_k tensor ops rather than O(B*L*top_k) Python steps.
    """
    def __init__(self, d_model, n_heads=8, factor=1, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)

    def time_delay_agg(self, values, corr):
        """
        Args:
            values: [B, H, D, L]
            corr:   [B, H, D, L]  (delay axis is the last dim)
        Returns:
            [B, H, D, L]
        """
        B, H, D, L = values.shape
        top_k = max(1, min(int(self.factor * np.log(L)), L))

        # delay index shared across the batch (Autoformer's speed-up design)
        mean_corr = corr.mean(dim=1).mean(dim=1)                    # [B, L]
        index = torch.topk(mean_corr.mean(dim=0), top_k, dim=-1)[1]  # [top_k]

        # per-sample weights on those delays
        weights = torch.softmax(mean_corr[:, index], dim=-1)        # [B, top_k]

        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i].item()), dims=-1)
            delays_agg = delays_agg + pattern * weights[:, i].view(B, 1, 1, 1)
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads

        # linear projections -> [B, H, L, D]
        queries = self.W_Q(queries).view(B, L, H, -1).transpose(1, 2)
        keys = self.W_K(keys).view(B, S, H, -1).transpose(1, 2)
        values = self.W_V(values).view(B, S, H, -1).transpose(1, 2)

        if S > L:
            values = values[:, :, :L, :]

        # period-based dependencies via FFT (Wiener-Khinchin), delay on last dim
        q_fft = torch.fft.rfft(queries.transpose(2, 3), dim=-1)     # [B,H,D,L//2+1]
        k_fft = torch.fft.rfft(keys.transpose(2, 3), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)  # [B,H,D,L]

        V = self.time_delay_agg(values.transpose(2, 3), corr)       # [B,H,D,L]
        V = V.transpose(2, 3).contiguous().view(B, L, -1)           # [B,L,H*D]

        return self.dropout(self.projection(V))


class AutoCorrelationBlock(nn.Module):
    """
    Encoder-style block built on Auto-Correlation (Autoformer, NeurIPS 2021).

    Autoformer's encoder applies SeriesDecomp after each sub-layer; here the
    decomposition is already handled by the STL module on the input side, so it
    is omitted to avoid having two decomposition mechanisms in the same model.
    What remains is the Auto-Correlation sub-layer plus a two-layer feed-forward
    network, each wrapped in a residual connection and layer normalisation.
    """
    def __init__(self, d_model, n_heads=8, factor=1, dropout=0.1, d_ff=None):
        super(AutoCorrelationBlock, self).__init__()
        d_ff = d_ff or d_model
        self.attn = AutoCorrelation(d_model, n_heads=n_heads, factor=factor,
                                    dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm1(x + self.dropout(self.attn(x, x, x)))
        return self.norm2(z + self.dropout(self.ffn(z)))


# ============================================================================
# 完整模型: TimesNet-Full
# ============================================================================

class Model(nn.Module):
    """
    TimesNet-Full: 完整论文方案
    
    架构流程:
    Input → STL分解 → Embedding → 
    ┌─ DualDomainTimesBlock (空间2D建模) ─┐
    │                                      ├→ 自适应融合 → LayerNorm → 
    └─ AutoCorrelation (时间依赖建模) ────┘
    × N layers → Projection → Output
    
    创新点:
    1. STL分解 (处理非平稳性)
    2. FFT+DCT双域周期检测 (互补性)
    3. 双路2D CNN (多尺度周期建模)
    4. AutoCorrelation (跨周期依赖)
    5. 并行双分支架构 + 自适应门控融合
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 1. STL分解模块
        self.use_stl = getattr(configs, 'use_stl', True)
        if self.use_stl:
            stl_period = getattr(configs, 'stl_period', 24)
            stl_impl = getattr(configs, 'stl_impl', 'torch')
            self.stl_decomposer = STLDecomposition(period=stl_period,
                                                   impl=stl_impl)
            # 三分量独立embedding
            self.trend_embedding = DataEmbedding(configs.enc_in, configs.d_model, 
                                                configs.embed, configs.freq, configs.dropout)
            self.seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)
            self.residual_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                              configs.embed, configs.freq, configs.dropout)
        
        # 2. DualDomainTimesBlock堆叠（并行分支1：空间周期建模）
        self.layer = configs.e_layers
        self.dual_blocks = nn.ModuleList([
            DualDomainTimesBlock(configs) for _ in range(configs.e_layers)
        ])
        
        # 3. AutoCorrelation层（并行分支2：时间依赖建模）
        self.use_autocorr = getattr(configs, 'use_autocorr', True)
        if self.use_autocorr:
            self.autocorr_layers = nn.ModuleList([
                AutoCorrelationBlock(configs.d_model, n_heads=8,
                                     factor=getattr(configs, 'autocorr_factor', 1),
                                     dropout=configs.dropout, d_ff=configs.d_ff)
                for _ in range(configs.e_layers)
            ])
            
            # 并行分支融合：两分支输出拼接后经全连接层
            self.branch_fusion = nn.ModuleList([
                nn.Linear(configs.d_model * 2, configs.d_model)
                for _ in range(configs.e_layers)
            ])
        
        # 4. 归一化和投影
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        长期预测任务
        
        Args:
            x_enc: [B, seq_len, enc_in] 输入序列
            x_mark_enc: [B, seq_len, mark_dim] 时间标记
            x_dec: [B, label_len+pred_len, dec_in] decoder输入
            x_mark_dec: [B, label_len+pred_len, mark_dim] decoder时间标记
        Returns:
            dec_out: [B, pred_len, c_out] 预测输出
        """
        # Normalization (from Non-stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # STL分解 + Embedding
        if self.use_stl:
            trend, seasonal, residual = self.stl_decomposer(x_enc)
            
            # 三分量独立embedding
            trend_emb = self.trend_embedding(trend, x_mark_enc)
            seasonal_emb = self.seasonal_embedding(seasonal, x_mark_enc)
            residual_emb = self.residual_embedding(residual, x_mark_enc)
            
            # 融合
            enc_out = trend_emb + seasonal_emb + residual_emb  # [B, T, d_model]
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 时序对齐
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 主干网络: 并行双分支架构
        for i in range(self.layer):
            # 分支1: DualDomainTimesBlock (空间2D周期建模)
            spatial_out = self.dual_blocks[i](enc_out)
            
            # 分支2: AutoCorrelation (时间依赖建模)
            if self.use_autocorr:
                temporal_out = self.autocorr_layers[i](enc_out)
                
                # 两分支输出沿特征维拼接后经全连接层融合
                concat_features = torch.cat([spatial_out, temporal_out], dim=-1)  # [B, T, 2*d_model]
                fused_out = self.branch_fusion[i](concat_features)                 # [B, T, d_model]

                # 残差连接
                enc_out = enc_out + fused_out
            else:
                # 如果不使用AutoCorrelation，只用空间分支
                enc_out = spatial_out
            
            # LayerNorm
            enc_out = self.layer_norm(enc_out)
        
        # 投影到输出空间
        dec_out = self.projection(enc_out)
        
        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        
        # 返回预测部分
        return dec_out[:, -self.pred_len:, :]
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
