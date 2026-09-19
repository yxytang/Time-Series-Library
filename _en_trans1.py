# -*- coding: utf-8 -*-
"""English translations, batch 1: title, abstract, introduction, 2.1."""

SIMPLE = {}

SIMPLE[0] = "A Parallel TimesNet Model with STL Decomposition and Dual-Domain Period Detection for Photovoltaic Power Forecasting"
SIMPLE[1] = "ABSTRACT"
SIMPLE[2] = (
    "PV power is intermittent and volatile, so accurate forecasting helps to reduce its impact on the power system. "
    "A PV power series is dominated by a daily cycle, and the power at the same time of day is strongly correlated across different dates; "
    "the accuracy of period modelling therefore directly determines the forecasting accuracy. "
    "To this end, this paper proposes the PA-STL-DualTimesNet model. "
    "First, STL decomposition splits the power series into trend, seasonal and residual components and reconstructs the input features, "
    "so that components differing greatly in magnitude and frequency obtain separate representations. "
    "Then, FFT and DCT period detection are carried out in parallel on the same input; their frequency grids are complementary, which reduces "
    "the effect of the grid discreteness of a single transform on two-dimensional folding, and each path performs its own folding and two-dimensional convolution. "
    "Meanwhile, an Autoformer encoder branch is introduced to aggregate, along the original time axis, the same-phase dependencies across cycles "
    "through the correlation spectrum, and its output is concatenated with that of the convolution branch and fused. "
    "On two real PV datasets, the proposed model is compared with eight baseline models, the contribution of the three modules is examined by ablation, "
    "and the results are grouped by season and daily volatility. "
    "The results show that the model outperforms all baselines on the four metrics, reducing the MAE by 8.2% and 8.6% relative to the runner-up TimesNet, "
    "and that the gain is more pronounced in seasons of stronger volatility."
)
SIMPLE[3] = "Keywords: photovoltaic power forecasting; STL decomposition; TimesNet; Autoformer encoder; parallel architecture"
SIMPLE[4] = "1. INTRODUCTION"
SIMPLE[5] = (
    "Solar energy has become one of the fastest-growing renewable energy sources because of its abundant reserves and flexible deployment [1]. "
    "Photovoltaic (PV) technology converts solar radiation into electricity, but its power is affected by weather fluctuations and is intermittent and stochastic, "
    "which brings challenges to grid stability and dispatch [2]. Accurate power forecasting is therefore of great significance for the stable operation "
    "and optimal dispatch of the grid."
)
SIMPLE[6] = (
    "According to the forecasting horizon, PV power forecasting can be divided into ultra-short-term, short-term and medium-to-long-term [3], "
    "and the power behaviour and modelling difficulties differ across these scales. From the viewpoint of its origin, however, PV power is always "
    "a superposition of two components: one determined by the position of the sun, which is strictly periodic with a phase fixed in the calendar "
    "and can therefore be extrapolated exactly; and one driven by weather processes, which appears as trend drift and random fluctuation and cannot be known in advance. "
    "This composition gives the dependence structure of the series a definite form rather than an arbitrary association between any two points: "
    "points of the same phase separated by an integer number of cycles (for example, the same time of day on different dates) are highly correlated, "
    "whereas the association between adjacent instants within one cycle is relatively weak. "
    "The key to modelling lies, therefore, in expressing explicitly the prior of “same phase across cycles” rather than long-sequence modelling ability "
    "in the general sense [4,5]."
)
SIMPLE[7] = (
    "PV power forecasting methods can be divided into physical, traditional statistical, artificial-intelligence and hybrid methods. "
    "Physical methods require a large number of meteorological and module parameters and are restricted by geographical conditions [6]. "
    "Traditional statistical methods (regression analysis, autoregressive moving average and grey theory) are good at capturing periodicity "
    "but struggle with nonlinear variation [7]. Machine-learning methods (random forest, support vector regression, XGBoost and LightGBM) improve "
    "the fitting of nonlinearity but model temporal dependence poorly [8]."
)
SIMPLE[8] = (
    "Deep learning remedies these shortcomings, and recent studies mostly adopt hybrid architectures that combine convolutional networks, "
    "recurrent networks and attention mechanisms so as to exploit the modelling capability of each [9]. For multi-step PV forecasting, existing work "
    "combines TimesNet with enhanced feature extraction and a dedicated loss function [10], or adopts a decompose–reconstruct–ensemble multi-scale "
    "framework to handle fluctuation [11]. Regarding decomposition strategies, recent years have also seen hybrid networks combining adaptive decomposition "
    "with physical-consistency constraints and secondary decomposition strategies [12,13], as well as combinations of variational mode decomposition "
    "with recurrent networks and attention mechanisms [14]. These works jointly indicate that explicitly separating trend and periodic components "
    "is an effective route to dealing with non-stationarity, but the manner of decomposition is itself still evolving [15]."
)
SIMPLE[9] = (
    "In period modelling, the idea of TimesNet is to fold the series along the period direction into a two-dimensional structure and then use "
    "two-dimensional convolution to capture both intra-period and inter-period variation [16], because a one-dimensional arrangement cannot explicitly "
    "express the association between points of the same phase in different cycles. The effectiveness of such methods depends on two premises: "
    "whether the period is estimated accurately, and whether the convolution kernel can cover enough cycles [17]. The former determines whether the rows "
    "remain aligned after folding, but existing implementations mostly rely on the spectral peak of a single transform, so once the true period falls "
    "between frequency grid points the detected period is biased and the two-dimensional tensor is progressively misaligned row by row. The latter bounds "
    "the range over which inter-period information can be aggregated: a convolution kernel covers only a limited neighbourhood, more distant same-phase "
    "dependencies must be transferred indirectly by stacking layers, and each layer re-estimating the period introduces further misalignment [18,19]. "
    "Complementary to this is the auto-correlation mechanism: sub-sequences at the same phase in different cycles are naturally similar, so a number of lags "
    "can be selected by delay similarity and the sub-sequences at the corresponding positions aligned as a whole and aggregated with weights [20]. "
    "It can span arbitrarily long lags without being constrained by kernel size, which exactly compensates for the shortcoming above; but it is usually "
    "tightly coupled with the encoder–decoder structure that introduced it and is difficult to embed in other models as an independent component [21,22]. "
    "In addition, some work models periodic components in the frequency domain or on frequency-enhanced representations, but such methods change the form "
    "of the representation and do not address the grid discreteness of period estimation itself [23]. At the decomposition level, although separating trend "
    "and periodic components has proved an effective way to mitigate non-stationarity, classical implementations rely on per-series iterative regression, "
    "which can neither be parallelised in batches nor differentiated and is therefore hard to embed in end-to-end training [24]."
)
SIMPLE[10] = (
    "In summary, existing work has the following shortcomings: (1) period detection mostly relies on a single transform and does not consider the effect "
    "of frequency-grid discreteness; (2) cross-cycle aggregation lacks a mechanism with clear long-lag semantics; and (3) decomposition modules are mostly "
    "implemented per series and are hard to parallelise in batches."
)
SIMPLE[11] = (
    "As for how models are combined, existing hybrid models mostly adopt a serial structure, in which a later module takes the output of an earlier one "
    "as its input. This is limited in the present task: if two-dimensional convolution and long-lag aggregation are connected in series, the input to "
    "auto-correlation is built on a representation that has already been folded and compressed, so the original lag structure it requires no longer exists. "
    "Parallel composition avoids this problem, but existing parallel schemes mostly place a convolution branch alongside a recurrent-network or attention branch; "
    "their improvement is concentrated on the aggregation side and is not combined with improvements in period detection. Consequently, the two shortcomings "
    "listed above — detection and aggregation — have not yet been addressed together in a single model. In response to these problems, the main "
    "contributions of this paper are as follows:"
)
SIMPLE[12] = (
    "(1) STL decomposition is used to decompose PV power into trend, seasonal and residual components, represented respectively by three embedding networks "
    "with independent parameters; a vectorised differentiable implementation replaces per-series LOESS so that the decomposition can be parallelised in batches."
)
SIMPLE[13] = (
    "(2) In period detection, DCT and FFT are introduced to form dual-domain detection; the difference between their frequency grids (2T/k and T/k) "
    "provides two complementary sets of period candidates."
)
SIMPLE[14] = (
    "(3) An Autoformer encoder layer is introduced as a parallel branch which, together with the TimesNet convolution branch, starts from the same hidden "
    "representation and takes charge of intra-period modelling and same-phase aggregation over arbitrarily long lags, respectively."
)
SIMPLE[15] = (
    "(4) Ablation, multi-model comparison and season-grouped experiments are carried out on two real PV datasets; the results show that the stronger "
    "the volatility, the greater the gain of the proposed model over the baselines."
)
SIMPLE[16] = "2. METHODOLOGY"
SIMPLE[17] = "2.1 STL decomposition"
SIMPLE[18] = "STL (Seasonal-Trend decomposition using Loess) decomposes a time series into three additive components — trend, seasonal and residual:"
SIMPLE[20] = (
    "Classical STL estimates the trend and seasonal terms by locally weighted regression (LOESS); its advantages are insensitivity to outliers and "
    "the ability to let seasonal strength vary over time. Its iterative solution, however, must be carried out per series, cannot be parallelised "
    "in batches and is not differentiable, so embedding it directly in end-to-end training incurs significant overhead. We therefore adopt a vectorised "
    "differentiable approximation: the trend term is estimated by a centred moving average (with the ends of the series replicated to remove boundary bias), "
    "and the seasonal term by averaging the detrended series across cycles phase by phase:"
)
SIMPLE[25] = (
    "Fig. 1 shows the STL decomposition of PV power for the two datasets. The trend component varies smoothly, reflecting slow drift at the seasonal scale; "
    "the seasonal component shows a stable daily waveform; and the residual component fluctuates about zero. Comparing the two columns, the residual "
    "component of the Xinjiang data has a larger amplitude, indicating stronger weather disturbance, which is consistent with the data characteristics "
    "described above."
)
SIMPLE[29] = (
    "The core characteristic of a PV power series is its strong periodicity: the alternation of day and night determines a daily cycle, and the evolution "
    "of weather systems superimposes longer cycles on top. How to represent this periodic structure explicitly in a model is the key to the task. "
    "TimesNet proposes a representation based on two-dimensional rearrangement for this purpose [25]; its overall structure is shown in Fig. 2."
)
SIMPLE[30] = (
    "Its starting point is the representational limitation of a one-dimensional arrangement. At every instant a time series exhibits two kinds of variation: "
    "intra-period variation with respect to adjacent instants, and inter-period variation with respect to points of the same phase in different cycles. "
    "A one-dimensional arrangement can express only the former explicitly: the adjacency of neighbouring samples is given directly by their indices, "
    "whereas the relation between points of the same phase in different cycles is folded into the index spacing and cannot be exploited directly "
    "by a local operator [26,27]."
)

# ---- paragraphs with inline equations -------------------------------------
# SEG[i] = (list of text segments, list of math indices in the order they appear)
# segments has len == len(math order) + 1
SEG = {}

SEG[23] = ([
    "where ",
    " is the trend window length, ",
    " the decomposition period, and ",
    " the number of cycles. The residual is the difference of the three, and a two-pass estimate (first removing the trend to obtain the seasonal term, "
    "then re-estimating the trend on the deseasonalised series) is used to reduce the correlation between components, corresponding to the inner loop of classical STL.",
], [0, 1, 2])

SEG[24] = ([
    "The decomposition period ",
    " is expressed in steps and must match the sampling interval; it must also satisfy ",
    ": the seasonal term relies on averaging across cycles and is meaningful only if at least two complete cycles are available, otherwise the seasonal term "
    "degenerates to zero and the decomposition module fails. This constraint directly affects high-frequency data and is discussed in detail in the experimental setup.",
], [0, 1])

SEG[31] = ([
    "The approach of TimesNet is first to rearrange the series into a two-dimensional tensor by period detection so that both kinds of variation gain locality. "
    "Given a series ",
    " of length ",
    " with ",
    " variables, the period is first determined from the FFT amplitude spectrum:",
], [2, 0, 1])

SEG[35] = ([
    "where the peak is taken only within ",
    " so as to exclude meaningless high-frequency noise. The series is then folded according to the period ",
    ":",
], [0, 1])

SEG[37] = ([
    "After rearrangement, the column direction of the tensor corresponds to adjacent instants and the row direction to points of the same phase differing by one cycle. "
    "Intra-period and inter-period variation are thus both converted into local patterns on the two-dimensional tensor and can be extracted simultaneously by the "
    "same convolution kernel [28]. TimesNet stacks TimesBlocks in a residual manner, re-estimating the period on the deep features at each layer, and aggregates "
    "the convolution results of the ",
    " periods with weights obtained by normalising the amplitudes through softmax:",
], [0])

SEG[40] = ([
    "where ",
    " is the multi-size two-dimensional convolution block.",
], [0])

SEG[46] = ([
    "The structure of this encoder layer is shown in Fig. 3, and its core is the auto-correlation mechanism. Unlike self-attention, which assigns weights "
    "according to the point-wise similarity of Query and Key, auto-correlation is based on the delay similarity of the series itself [31]. For a real discrete "
    "process ",
    ", the auto-correlation at lag ",
    " is defined as",
], [0, 1])

SEG[48] = ([
    "It describes the degree of similarity between the series and its version lagged by ",
    " steps. The rationale is that sub-processes at the same phase position in different cycles are naturally similar, so ",
    " takes significant values when ",
    " is an integer multiple of the period and can be regarded as an unnormalised confidence of the period length ",
    ". A direct computation by definition costs ",
    ", which the Wiener–Khinchin theorem reduces to ",
    ":",
], [0, 1, 2, 3, 4, 5])

SEG[51] = ([
    "Let ",
    " be linearly projected to obtain ",
    ", ",
    " and ",
    ". The mechanism first selects the k delays with the highest confidence and then, using their normalised confidences as weights, performs time-delay "
    "aggregation of the Value at each delay:",
], [0, 1, 2, 3])

SEG[55] = ([
    "where ",
    " shifts ",
    " by ",
    " steps along the time axis and ",
    " with c a hyper-parameter. This aggregation differs fundamentally from the point-wise weighting of self-attention: it aligns as a whole the similar "
    "sub-sequences at the same phase and then sums them with weights, so when the delay happens to be the true period, what is aggregated is precisely "
    "the same-phase points separated by an integer number of cycles. Information can therefore cross arbitrarily long lags in a single operation, without "
    "the layer-by-layer accumulation required by convolution, and the gap left by the structure described above is thereby filled.",
], [0, 1, 2, 3])

SEG[64] = ([
    "TimesNet determines the period from the peak of the FFT amplitude spectrum. For a series of length ",
    ", the FFT is",
], [0])

SEG[66] = ([
    "Because the spectrum takes values only at the discrete frequencies ",
    ", the detectable periods are restricted to",
], [0])

SEG[68] = ([
    "this set of grid points. When the true period ",
    " does not fall on a grid point, a deviation exists between the detected ",
    " and ",
    ", and this deviation makes the rows of the two-dimensional tensor shift one by one and weakens inter-period locality. The grid is particularly sparse "
    "at long periods: the spacing between adjacent points is about ",
    ", so the resolution falls as the period grows.",
], [0, 1, 2, 3])

SEG[71] = ([
    "Compared with FFT, the frequency index differs by a factor of two: the ",
    " -th basis contains only ",
    " complete cycles within the window, so the detectable periods are ",
    ". Expanding by the parity of ",
    " gives: when ",
    " is even, ",
    " coincides with the FFT grid; when ",
    " is odd, it gives ",
    ", which lies midway between adjacent FFT grid points. The DCT grid is therefore a refinement of the FFT grid: it retains every period representable "
    "by FFT and additionally provides candidates at half-interval positions. If the true period falls between two FFT grid points, DCT may yield an estimate "
    "closer to ",
    ", thereby improving row alignment.",
], [0, 1, 2, 3, 4, 5, 6, 7, 8])

SEG[78] = ([
    "where ",
    " and ",
    " are the weight matrix and bias of this fully connected layer, respectively.",
], [0, 1])

SEG[79] = ([
    "Parallel rather than serial composition is adopted because the two branches require different inputs: the encoder branch depends on the lag semantics "
    "of the original time axis, whereas two-dimensional folding is a rearrangement that destroys those semantics, so if it followed the convolution branch "
    "the resulting ",
    " would no longer correspond to a real time delay; conversely, the convolution branch depends on the phase alignment produced by folding, which would "
    "likewise be destroyed if it followed auto-correlation. Each branch starting directly from the same hidden representation allows the two kinds of "
    "inter-period aggregation to be carried out in their own meaningful coordinate systems. Their failure conditions also differ: the convolution branch "
    "is affected by period-estimation error, whereas the encoder branch is affected by the quality of the correlation spectrum; parallel composition lets "
    "the model fall back on one when the other is unreliable, which is why the fusion weights are left to be learned by the network itself.",
], [0])

SEG[113] = ([
    "where ",
    " and ",
    " are the true and predicted values respectively, ",
    " is the number of samples and ",
    " the mean of the true values. The denominator of MASE is the mean absolute error of the seasonal naive forecast (one daily cycle apart); "
    "a value below 1 means better than this baseline and above 1 worse.",
], [0, 1, 2, 3])
