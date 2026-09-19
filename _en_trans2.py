# -*- coding: utf-8 -*-
"""English translations, batch 2: 2.2 onwards to the references."""

SIMPLE = {}

SIMPLE[41] = (
    "The effectiveness of the structure above depends on the accuracy of period estimation and the coverage of the convolution kernel [29]; "
    "the two improvements that follow target exactly these two points."
)
SIMPLE[44] = "2.3 Autoformer encoder branch"
SIMPLE[45] = (
    "TimesNet aggregates inter-period information along the row direction by two-dimensional convolution, but the coverage of the convolution kernel is limited. "
    "The largest kernel in the Inception block has size 11, covering only about ±5 cycles on the folded tensor; more distant same-phase dependencies can only be "
    "transferred indirectly by stacking layers, and every temporal block re-estimates the period, which shifts the row alignment after folding, so stacking does "
    "not reliably widen this range. Enlarging the kernel makes the parameter count grow with the square of the kernel size, while deepening the network is disturbed "
    "by the re-estimated period. The convolution branch therefore lacks a scalable way of aggregating same-phase information over long distances. To retain "
    "intra-period modelling capability while obtaining long-lag aggregation unrestricted by kernel size, we introduce an Autoformer encoder layer as the second "
    "parallel branch [30]."
)
SIMPLE[56] = (
    "The original Autoformer encoder applies a series decomposition after each sub-layer in order to separate and discard the trend component [32]. In this paper "
    "the decomposition is already performed by the STL module at the input, so to avoid two decomposition mechanisms within one model it is not repeated here; "
    "only the auto-correlation sub-layer and the feed-forward network are retained, each with a residual connection and layer normalisation:"
)
SIMPLE[59] = "where the feed-forward network is a two-layer fully connected structure."
SIMPLE[62] = "2.4 FFT-DCT dual-domain period detection"
SIMPLE[63] = "The accuracy of period estimation directly determines whether the two-dimensional structure holds, so period detection needs to be examined separately."
SIMPLE[69] = "To remedy this, a second set of period grids is introduced. DCT-II uses a cosine basis with a half-sample offset:"
SIMPLE[72] = (
    "Accordingly, two detection paths are run in parallel on the same input: the FFT path takes the Top-K periods, and the DCT path applies DCT-II and likewise "
    "takes the Top-K peaks. Each path performs folding and convolution with its own set of periods, the parameters are not shared, and the outputs are concatenated "
    "along the feature dimension and fused by a fully connected layer."
)
SIMPLE[73] = (
    "A refined grid is valuable only if the additional candidates really are true periods; if they merely correspond to noise peaks in the amplitude spectrum, "
    "their folded rows do not form a same-phase relation and instead introduce noise. This paper therefore does not presuppose that DCT brings a gain, but treats "
    "it as a testable hypothesis to be decided by ablation."
)
SIMPLE[74] = "2.5 PA-STL-DualTimesNet model"
SIMPLE[75] = (
    "The overall structure of the proposed parallel STL dual-domain TimesNet (PA-STL-DualTimesNet) is shown in Fig. 4 and consists of the following parts. First, "
    "STL decomposes PV power into trend, seasonal and residual components, and the three components are mapped by embedding networks with independent parameters "
    "and then added to form the reconstructed input features. The backbone is then formed by stacking N dual-domain temporal blocks (N is 2 by default)."
)
SIMPLE[76] = (
    "Two branches are set in parallel within each block: the dual-domain periodic convolution branch extracts intra-period and near-neighbour inter-period variation "
    "by period folding and two-dimensional convolution, while the Autoformer encoder branch selects delays from the correlation spectrum and aggregates same-phase "
    "points over arbitrarily long lags along the original time axis. The outputs of the two branches are concatenated along the feature dimension, passed through "
    "a fully connected layer to map back to the original dimension, added to the input residual and normalised:"
)
SIMPLE[80] = (
    "It should be noted that “parallel” here means that both branches take the same hidden representation as input and are not data-dependent on each other. "
    "This is not on the same level as the parallel computation represented by Informer (which describes whether the computation can be parallelised) or the "
    "multi-period parallelism represented by TimesNet (which describes multi-scale processing)."
)
SIMPLE[81] = "Finally, the output of the backbone is projected into the forecasting space by a fully connected layer and denormalised to give the final power forecast."
SIMPLE[84] = "3. DATA AND EXPERIMENTAL SETUP"
SIMPLE[85] = "3.1 Data description"
SIMPLE[86] = (
    "Two real PV datasets are used to evaluate the model. They come from Australia and China respectively and differ markedly in climate and volatility, which "
    "provides a natural contrast for testing the model under different conditions."
)
SIMPLE[87] = (
    "The first dataset comes from the DKA Solar Centre in Darwin, Northern Territory, Australia (denoted Solar-DKA). The centre publishes measured data for several "
    "PV arrays, of which the multivariate power and meteorological records are used here. The sampling interval is 5 minutes, the time span is January 2014 to "
    "November 2015, and there are 192,375 records in total. The data contain 10 numerical variables: the generated power Active_Power (the forecasting target) "
    "and nine meteorological variables — wind speed, air temperature, relative humidity, global horizontal radiation, diffuse horizontal radiation, wind direction, "
    "daily rainfall, global tilted radiation and diffuse tilted radiation. The area has a tropical monsoon climate with high temperatures and abundant sunshine "
    "throughout the year, and the generated power is relatively stable overall."
)
SIMPLE[88] = (
    "The second dataset comes from a PV plant in the Xinjiang Uygur Autonomous Region of China (denoted Solar-XJ). The sampling interval is 15 minutes, the time "
    "span is the whole of 2023, and there are 35,040 records with no missing values. The data contain 8 numerical variables: the actual generated power (the "
    "forecasting target) and seven meteorological variables — module temperature, ambient temperature, pressure, humidity, global radiation, direct radiation and "
    "diffuse radiation. The area has a temperate continental climate with a large diurnal temperature range and frequent weather changes."
)
SIMPLE[89] = (
    "Compared with the first dataset, the second has a markedly higher proportion of days with pronounced power fluctuation. This difference is precisely the "
    "motivation for the season- and volatility-grouped experiments: if the relative gain of the model is larger under volatile conditions, its improvement indeed "
    "comes from more robust modelling of the periodic structure and the long-lag dependence, rather than being effective only under stable conditions. Table 1 "
    "gives the detailed characteristics of the two datasets."
)
SIMPLE[91] = "3.2 Data preprocessing"
SIMPLE[92] = (
    "To make the two datasets comparable, the 5-minute Solar-DKA data are first resampled to 15 minutes, so that both have the same sampling interval and a daily "
    "cycle of 96 samples, eliminating any bias caused by the difference in sampling frequency."
)
SIMPLE[93] = (
    "For data cleaning, PV power is physically non-negative and should not exceed the installed capacity; the 3σ rule is therefore used to identify outliers, which "
    "are set to missing. Short gaps are filled by linear interpolation, and intervals with missing data longer than 3 hours are removed entirely — such intervals "
    "usually correspond to equipment downtime, are hard to impute reliably, and forcing a fill would introduce spurious periodic structure."
)
SIMPLE[94] = (
    "For the features, the maximal information coefficient (MIC) is used to measure the correlation between each meteorological variable and the generated power. "
    "MIC captures both linear and nonlinear dependence and is suitable for a complex relationship such as irradiance versus power. The results are shown in Fig. 5: "
    "in both datasets the irradiance variables (global, direct and diffuse radiation) have MIC values clearly higher than the others and are the dominant inputs; "
    "module temperature shows a moderate correlation; and pressure, wind direction and rainfall have the lowest MIC. Variables with MIC below 0.1 are therefore "
    "removed to reduce the input dimension and the risk of overfitting. Comparing the two datasets, the correlations among variables are generally higher in the "
    "Xinjiang data, indicating that the power at this site is more strongly affected by several factors jointly and that a single variable has weaker explanatory power."
)
SIMPLE[97] = (
    "Finally, the power and meteorological variables are normalised to zero mean and unit variance, with the statistics fitted on the training set only and then "
    "applied to the validation and test sets; the data are split chronologically into training, validation and test sets in a 6:2:2 ratio without shuffling, so as "
    "to avoid future-information leakage. The validation set is used for early stopping and model selection, and the test set only for final evaluation. The datasets "
    "retain all periods and no daytime filtering is applied. Night-time power is constantly zero, which does dilute the error signal during the day, but filtering "
    "the night would make “one day” no longer 96 samples and destroy the correspondence between the daily cycle and the number of steps. To avoid the effect of the "
    "night-time zeros, the evaluation also reports R² and MASE, which are not affected by zeros, in addition to RMSE and MAE."
)
SIMPLE[98] = (
    "The STL decomposition period is taken as one daily cycle, i.e. p = 96 steps; the constraint T ≥ 2p requires the input length to be no less than 192 steps."
)
SIMPLE[99] = "3.3 Baseline models"
SIMPLE[100] = (
    "To evaluate the performance of the proposed model, eight models are selected for comparison, covering four families of methods: machine learning, recurrent "
    "networks, linear models and Transformers."
)
SIMPLE[101] = (
    "The machine-learning methods are K-nearest neighbours (KNN) and LightGBM; both are simple to construct and fast to train and serve as a reference for "
    "assessing the need for deep models. The recurrent networks are LSTM and BiLSTM, the former a classical method of time-series modelling and the latter "
    "aggregating context in both directions. DLinear consists of a linear decomposition plus a multilayer perceptron and is a widely cited simple strong baseline "
    "used to test the need for complex structures. PatchTST splits the series into patches and models them with a Transformer, representing the patch-based approach. "
    "Autoformer introduces the auto-correlation mechanism and series decomposition and is the source of this paper's encoder branch; including it in the comparison "
    "shows directly the difference between “borrowing its component” and “using the model as it is”. TimesNet is the model this paper improves directly and is the "
    "main baseline."
)
SIMPLE[102] = (
    "All baselines are trained and evaluated under the same data split, the same input and output lengths and the same normalisation procedure, so that the "
    "comparison is fair. The hyper-parameter configuration of the proposed model is given in Table 2."
)
SIMPLE[104] = (
    "The loss during training is shown in Fig. 6. The training and validation losses fall rapidly in the first few epochs and then level off; the validation loss "
    "reaches its minimum in the middle of training and does not improve afterwards, at which point early stopping terminates training. The gap between the two "
    "curves is small and no obvious overfitting occurs."
)
SIMPLE[107] = "3.4 Evaluation metrics"
SIMPLE[108] = (
    "To evaluate the model on the test set, four metrics are adopted — mean absolute error (MAE), root mean square error (RMSE), mean absolute scaled error (MASE) "
    "and the coefficient of determination (R²) — defined as follows:"
)
SIMPLE[114] = "4. RESULTS AND DISCUSSION"
SIMPLE[115] = "4.1 Comparison with different models"
SIMPLE[116] = "The PA-STL-DualTimesNet model is compared with the baselines described above; the results for the two datasets are given in Table 3 and Table 4."
SIMPLE[119] = (
    "The comparison shows that machine-learning models are limited in their ability to capture temporal dependence and have clearly higher errors than the deep "
    "models; recurrent networks outperform the machine-learning models but still model long lags inadequately; DLinear reaches a level close to the deep models "
    "with a very simple structure, which shows that the baselines themselves are strong; PatchTST and Autoformer perform similarly, and the latter — the source of "
    "this paper's encoder branch — is less accurate than the proposed model, which shows that decoupling the auto-correlation sub-layer from the complete model and "
    "combining it in parallel with the period-folding path is effective. The proposed model outperforms all baselines on the four metrics, and the conclusion is "
    "consistent across the two datasets."
)
SIMPLE[120] = (
    "Fig. 7 shows the MAE of each model as a bar chart. The ranking of the models is exactly the same for the two datasets, the proposed model being lowest in both, "
    "which agrees with the conclusions of Tables 3 and 4 and shows that the advantage is not incidental."
)
SIMPLE[123] = (
    "An overall error can only reflect average performance, so Fig. 8 gives the prediction curves on some test samples (four days, 384 sampling points in total). "
    "All models capture the daily waveform of the true power, and the differences are concentrated in the periods of rapid power ramping: there the baselines deviate "
    "noticeably more, while the proposed model tracks the true values most closely. Fig. 9 gives the same judgement from the point-by-point perspective. The scatter "
    "of the proposed model lies closer to the diagonal; in the high-power region the baselines generally underestimate the peak power, whereas the proposed model "
    "has a smaller deviation."
)
SIMPLE[129] = (
    "Fig. 10 shows the distribution of the absolute prediction error of each model as a box plot: the upper and lower edges of the box are the 25th and 75th "
    "percentiles of the error, the horizontal line is the median, the whiskers extend to 1.5 times the interquartile range, and the circles are outliers beyond "
    "that range. Compared with a mean metric, a box plot reflects the dispersion of the error — the box of the proposed model is both lower and narrower overall, "
    "showing that it is not only better in average accuracy but also less variable in error. Some baselines have a similar median but a longer box and more outliers, "
    "and their error is clearly larger in individual periods."
)
SIMPLE[132] = "4.2 Ablation study"
SIMPLE[133] = (
    "To verify the independent contribution of each module, six ablation variants are constructed on the TimesNet baseline by introducing STL decomposition, "
    "FFT-DCT dual-domain period detection and the Autoformer encoder branch separately, and each is evaluated on both datasets; the results are given in Tables 5 and 6."
)
SIMPLE[137] = (
    "The ablation variants are constructed by switching the three modules off individually (controlled by --use_stl, --use_dct and --use_autocorr in the "
    "implementation); the DCT branch can be switched off on its own, so “FFT only” and “FFT + DCT” can be compared directly. It should be noted that switching off "
    "DCT nearly halves the parameter count (measured, about 9.47 M down to 4.76 M), so the parameter difference must be stated whenever the dual-domain gain is "
    "reported; otherwise it is hard to distinguish an information gain from a capacity gain. Moreover, the individual gains of the modules are not additive, and "
    "interaction effects should be judged from measured pairwise combinations rather than by simple addition."
)
SIMPLE[138] = (
    "Fig. 11 shows the MAE of each ablation configuration as a bar chart. As the modules are introduced one by one the error falls, the full combination giving the "
    "lowest value; the trend is the same for both datasets, indicating that the contribution of each module is stable."
)
SIMPLE[141] = "4.3 Seasonal and volatility analysis"
SIMPLE[142] = (
    "To examine the performance of the model under strongly fluctuating conditions, the test set is grouped by month. The representative month of each quarter "
    "(April, July, October and January) is selected, with the volatility of that month's data as a reference; the results are given in Table 7."
)
SIMPLE[144] = (
    "Both datasets show the same pattern: the stronger the volatility, the greater the gain of the proposed model over the baseline. Spring (dusty winds, frequent "
    "synoptic changes) gives the largest improvement, 14% and 13% for the two datasets; summer (a high proportion of clear, stable weather) gives the smallest, only "
    "about 7%. This shows that the model performs more conspicuously in highly fluctuating periods."
)
SIMPLE[145] = (
    "This pattern can be explained by the mechanism of the three modules. In period detection, stronger fluctuation raises the spectral side-lobes and relatively "
    "weakens the main peak, so a single transform is more likely to miss the period or shift the peak, and an erroneous period estimate directly destroys the phase "
    "alignment of the two-dimensional folding; the two candidate sets given by FFT and DCT with different grids are then more valuable. In cross-cycle aggregation, "
    "the reference value of “the same time yesterday” falls under fluctuating weather and the model must rely on periodic priors at longer time scales, so the encoder "
    "branch's long-lag aggregation, unconstrained by kernel size, contributes more. In input decomposition, stronger volatility means a larger share of residual energy "
    "and makes the trend and seasonal components easier to mask by random disturbance; after STL separates the three, each component is represented by independent "
    "parameters and is therefore more robust to non-stationarity."
)
SIMPLE[146] = "4.4 Period detection analysis"
SIMPLE[147] = (
    "To test whether the dual-domain detection really is complementary, the Top-5 periods detected by the two paths are derived separately from the raw series; "
    "the results are given in Table 8."
)
SIMPLE[149] = (
    "The two sets of detected periods differ clearly, the intersection containing only the daily cycle and its half-frequency (96 and 48), which shows that "
    "dual-domain detection is not a repeated extraction of the same information."
)
SIMPLE[150] = (
    "The origin of this difference needs to be distinguished. The periods detected by FFT are all integer harmonics of the daily cycle (96/1, 96/2, 96/3, 96/4, 96/6), "
    "consistent with the physical origin of PV power; whereas the extra periods found by DCT (32 h, 19 h, 13.5 h, etc.) are not harmonics of the daily cycle and have "
    "no counterpart in the physics of PV power. Since the period grid spacing of DCT is half that of FFT, these additional candidates are more likely to be spurious "
    "peaks produced by the finer grid falling at non-harmonic positions than real periods drowned out by FFT."
)
SIMPLE[151] = (
    "Accordingly, dual-domain detection is positioned as follows: the two paths provide different sets of period candidates, FFT giving harmonics consistent with the "
    "physical origin and DCT supplementing candidates on a finer grid; whether DCT yields a net gain depends on whether these candidates form an effective intra-period "
    "structure after two-dimensional folding, which must be decided by ablation and cannot be inferred from the difference between the period sets alone. On the "
    "available evidence, the conclusion of this paper is limited to the fact that the two paths give different candidate sets (supported by Table 8); it is not "
    "sufficient to show that DCT recovers real periods missed by FFT."
)
SIMPLE[152] = (
    "One implementation detail should also be noted: the comparison above is made on the raw series in order to match physical periods, whereas period detection in "
    "the model actually acts on the hidden representation after decomposition and embedding. On that hidden representation the periods detected by both paths are "
    "concentrated at 2–4 steps and do not correspond directly to the daily cycle, so Table 8 reflects the difference in resolving power of the two transforms on the "
    "raw signal, while the folding shape inside the model is determined by the detection result on the hidden representation; the two should be viewed separately. "
    "Looking further, these two kinds of period belong to different levels of representation: the 2–4 steps on the hidden representation is the scale of the folding "
    "geometry inside the network and determines the row and column partition of the two-dimensional tensor, whereas the 96 steps on the raw series is the physical "
    "period of the power and is used only for comparison with Table 8. Precisely because the period on the hidden representation is short, the input window still "
    "contains several tens of cycles after folding, which is what makes the earlier judgement about the limited coverage of the convolution kernel valid; if folding "
    "were done at the daily period, the window would contain only two cycles and that limitation would not arise."
)
SIMPLE[153] = "5. CONCLUSION"
SIMPLE[154] = (
    "This paper addresses PV power forecasting with the PA-STL-DualTimesNet model. At the input, STL decomposition separates the power series into trend, seasonal "
    "and residual components that are embedded separately, avoiding mutual interference between components whose magnitude and frequency differ greatly. At period "
    "detection, the different frequency grids of FFT and DCT (T/k and 2T/k respectively) are used to construct two complementary sets of period candidates, so that "
    "detection no longer depends on a single grid. At aggregation, a parallel structure lets the two-dimensional convolution branch and the Autoformer encoder branch "
    "start directly from the same hidden representation and capture intra-period local shape and cross-cycle long-lag dependence respectively, which are then fused by "
    "concatenation and a fully connected layer. Parallel rather than serial composition is used because auto-correlation depends on the lag structure of the original "
    "time axis while convolution depends on the phase alignment after folding, and the two requirements are mutually incompatible. On the two real PV datasets from "
    "DKA in Australia and Xinjiang in China, the proposed model outperforms machine-learning, recurrent-network, hybrid and recent representative methods on all four "
    "metrics, MAE, RMSE, MASE and R². Further monthly grouping shows that the stronger the volatility, the greater the gain over the baseline (from about 7% in "
    "low-volatility months to about 13% in high-volatility months), indicating that the improvement comes from more robust modelling of the periodic structure and "
    "the long-lag dependence rather than from a general gain in capacity."
)
SIMPLE[155] = "References"

SEG = {}
