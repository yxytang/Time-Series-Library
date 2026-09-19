"""
训练脚本: TimesNet-Full 完整论文方案
集成STL + FFT+DCT + AutoCorrelation

使用方法:
python train_solar_full.py --model TimesNetFull --data solar_dka --pred_len 96
"""

import argparse
import random
import sys
import os

import numpy as np
import pandas as pd
import torch

# Windows consoles default to GBK, which raises on the emoji in the banner.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

_KNOWN_TIME_COLS = ('timestamp', 'date', 'time', 'datetime', '时间', '日期', '时间戳')


def infer_num_vars(root_path, data_path):
    """数值变量数 = 总列数 - 时间列数（两个数据集列名不同，故自动识别）。"""
    fp = os.path.join(root_path, data_path)
    df = pd.read_csv(fp, nrows=5) if fp.endswith('.csv') else pd.read_excel(fp, nrows=5)
    n_time = sum(1 for c in df.columns if str(c).strip().lower() in _KNOWN_TIME_COLS)
    return len(df.columns) - (n_time if n_time else 1)


def main():
    # 修复随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet-Full for Solar Forecasting')

    # ======================== 基础配置 ========================
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1,
                        help='status')
    parser.add_argument('--model_id', type=str, default='solar_96',
                        help='model id')
    parser.add_argument('--model', type=str, default='TimesNetFull',
                        help='model name')

    # ======================== 数据配置 ========================
    parser.add_argument('--data', type=str, default='solar_dka',
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data_provider/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='DKA Solar Center dataset.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate, S:univariate, MS:multivariate to univariate')
    parser.add_argument('--target', type=str, default='Active_Power',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')

    # ======================== 时序配置 ========================
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='subset for M4')

    # ======================== TimesNet-Full 特有配置 ========================
    
    # STL配置
    parser.add_argument('--use_stl', type=int, default=1,
                        help='whether to use STL decomposition')
    parser.add_argument('--stl_period', type=int, default=24,
                        help='STL seasonal period in STEPS. DKA is 5-min sampled, '
                             'so a daily cycle is 288 steps; 24 steps = 2h')
    parser.add_argument('--stl_impl', type=str, default='torch',
                        choices=['torch', 'statsmodels'],
                        help="'torch' = vectorized/differentiable (fast); "
                             "'statsmodels' = original LOESS STL (slow, CPU)")
    
    # 双域配置
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k periods for FFT and DCT')
    parser.add_argument('--use_dct', type=int, default=1,
                        help='whether to use the DCT branch (0 = FFT-only ablation)')
    
    # AutoCorrelation配置
    parser.add_argument('--use_autocorr', type=int, default=1,
                        help='whether to use AutoCorrelation')
    parser.add_argument('--autocorr_factor', type=int, default=1,
                        help='factor for AutoCorrelation time delay aggregation')
    
    # ======================== 模型配置 ========================
    parser.add_argument('--enc_in', type=int, default=0,
                        help='encoder input size; 0 = auto-detect from the data file')
    parser.add_argument('--dec_in', type=int, default=0,
                        help='decoder input size; 0 = auto-detect')
    parser.add_argument('--c_out', type=int, default=0,
                        help='output size; 0 = auto-detect')
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='num of heads for AutoCorrelation')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64,
                        help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1,
                        help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in encoder')
    
    # TimesNet特有参数
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='number of kernels for Inception')

    # ======================== 优化配置 ========================
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training',
                        default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='use gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus',
                        default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multile gpus')

    args = parser.parse_args()

    # ======================== 输入/输出维度自适应 ========================
    # Solar-DKA 有10个数值列，site1.xlsx 有8个；列名还不同（中文/英文），
    # 因此默认从数据文件推断，避免与数据维度不匹配。
    if args.enc_in <= 0:
        args.enc_in = infer_num_vars(args.root_path, args.data_path)
    if args.dec_in <= 0:
        args.dec_in = args.enc_in
    if args.c_out <= 0:
        args.c_out = args.enc_in

    # ======================== GPU配置 ========================
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('='*50)
    print('🚀 TimesNet-Full 训练开始')
    print('='*50)
    print(f'📊 数据集: {args.data}  ({args.data_path})')
    print(f'📐 变量数: {args.enc_in}  (enc_in=dec_in=c_out)')
    print(f'📏 输入长度: {args.seq_len}, 预测长度: {args.pred_len}')
    print(f'🔧 模型配置:')
    print(f'   - d_model: {args.d_model}')
    print(f'   - e_layers: {args.e_layers}')
    print(f'   - top_k: {args.top_k}')
    print(f'🎯 创新模块:')
    print(f'   - STL分解: {"✅" if args.use_stl else "❌"} (period={args.stl_period}, impl={args.stl_impl})')
    print(f'   - FFT+DCT双域: {"✅" if args.use_dct else "❌仅FFT"} (top_k={args.top_k})')
    print(f'   - AutoCorrelation: {"✅" if args.use_autocorr else "❌"}')
    print(f'💻 设备: {"GPU " + str(args.gpu) if args.use_gpu else "CPU"}')
    print('='*50)

    # ======================== 多次实验 ========================
    for ii in range(args.itr):
        # 设置实验
        setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'

        exp = Exp_Long_Term_Forecast(args)  # 设置实验

        # 训练
        print(f'\n>>>>>> 开始训练 : {setting} >>>>>>')
        exp.train(setting)

        # 测试
        print(f'\n>>>>>> 开始测试 : {setting} >>>>>>')
        exp.test(setting)
        torch.cuda.empty_cache()

    print('\n' + '='*50)
    print('✅ 训练完成！')
    print('='*50)


if __name__ == "__main__":
    main()
