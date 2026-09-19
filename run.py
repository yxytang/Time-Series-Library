import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet for Solar Power Forecasting')

    # ============ 基础配置 ============
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='任务类型: long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='训练状态: 1=训练+测试, 0=仅测试')
    parser.add_argument('--model_id', type=str, default='solar_forecast', help='实验标识')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='模型名称: TimesNet, TimesNetPlus')

    # ============ 数据配置 ============
    parser.add_argument('--data', type=str, default='solar_dka',
                        help='数据集: solar_dka, solar_site1, custom')
    parser.add_argument('--root_path', type=str, default='./data_provider/',
                        help='数据文件根目录')
    parser.add_argument('--data_path', type=str, default='DKA Solar Center dataset.csv',
                        help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务: M=多变量预测多变量, S=单变量预测单变量, MS=多变量预测单变量')
    parser.add_argument('--target', type=str, default='Active_Power', help='目标列名')
    parser.add_argument('--freq', type=str, default='t',
                        help='时间特征编码: s=秒, t=分钟, h=小时, d=天, b=工作日, w=周, m=月')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存路径')

    # ============ 预测任务配置 ============
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度 (5min间隔: 96=8小时)')
    parser.add_argument('--label_len', type=int, default=48, help='解码器起始token长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度 (5min间隔: 96=8小时)')
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='季节性模式')

    # ============ TimesNet模型配置 ============
    parser.add_argument('--top_k', type=int, default=5, help='FFT周期检测的top-k个数')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception block的卷积核数量')
    parser.add_argument('--enc_in', type=int, default=10, help='编码器输入维度 (特征数)')
    parser.add_argument('--dec_in', type=int, default=10, help='解码器输入维度')
    parser.add_argument('--c_out', type=int, default=10, help='输出维度')
    parser.add_argument('--d_model', type=int, default=128, help='模型隐藏层维度')
    parser.add_argument('--e_layers', type=int, default=2, help='TimesBlock层数')
    parser.add_argument('--d_ff', type=int, default=256, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')

    # ============ 优化器配置 ============
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--train_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练', default=False)

    # ============ GPU配置 ============
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用GPU')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='禁用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU类型: cuda, mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备列表')

    args = parser.parse_args()

    # 设备配置
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: cuda:{args.gpu}')
    else:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = torch.device('mps')
            print('Using MPS (Apple Silicon GPU)')
        else:
            args.device = torch.device('cpu')
            print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('=' * 60)
    print('实验配置:')
    print_args(args)
    print('=' * 60)

    # ============ 导入实验类 ============
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # 实验设置标识
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_ff,
                args.embed,
                args.des,
                ii)

            print('>' * 60)
            print(f'开始训练: {setting}')
            print('>' * 60)

            exp = Exp(args)
            exp.train(setting)

            print('>' * 60)
            print(f'开始测试: {setting}')
            print('>' * 60)
            exp.test(setting)

            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        # 仅测试模式
        exp = Exp(args)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.embed,
            args.des,
            ii)

        print('>' * 60)
        print(f'加载模型进行测试: {setting}')
        print('>' * 60)
        exp.test(setting, test=1)

        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
