import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


class Dataset_Solar(Dataset):
    """
    太阳能数据集加载器，支持CSV和Excel格式
    数据包含：timestamp, Active_Power, 以及多个气象特征
    """
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='DKA Solar Center dataset.csv',
                 target='Active_Power', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 96   # 默认96个时间步 (8小时 @ 5min间隔)
            self.label_len = 48  # 48个时间步
            self.pred_len = 96   # 预测96个时间步
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)

        # 支持CSV和Excel格式
        if local_fp.endswith('.csv'):
            df_raw = pd.read_csv(local_fp)
        elif local_fp.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(local_fp)
        else:
            raise ValueError(f"Unsupported file format: {local_fp}")

        '''
        df_raw.columns: [<time column>, ...(feature columns)]
        时间列名可能是 timestamp / 时间 / date 等，需自动识别。
        '''
        # 识别时间列：先按常见名字，再退回"前几列中能解析为日期的那一列"
        time_col = None
        _known = ('timestamp', 'date', 'time', 'datetime', '时间', '日期', '时间戳')
        for c in df_raw.columns:
            if str(c).strip().lower() in _known:
                time_col = c
                break
        if time_col is None:
            for c in df_raw.columns[:3]:
                try:
                    pd.to_datetime(df_raw[c])
                    time_col = c
                    break
                except (ValueError, TypeError):
                    continue
        if time_col is None:
            raise ValueError(f'无法识别时间列，列名: {list(df_raw.columns)}')
        df_raw = df_raw.rename(columns={time_col: 'timestamp'})
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

        # 获取所有数值列（除了timestamp）
        cols = [c for c in df_raw.columns if c != 'timestamp']

        # 确保目标列存在：否则优先选含 power/功率/发电 的列，再退回首列
        if self.target not in cols:
            cand = [c for c in cols
                    if any(k in str(c).lower() for k in ('power', '功率', '发电', 'active'))]
            self.target = cand[0] if cand else cols[0]

        # 重新排列列顺序：timestamp + 特征 + 目标
        cols.remove(self.target)
        df_raw = df_raw[['timestamp'] + cols + [self.target]]

        # 数据集划分（按时间顺序）
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 除了timestamp的所有列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        # 时间特征编码
        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp['timestamp'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['timestamp'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['timestamp'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['timestamp'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['timestamp'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['timestamp'].apply(lambda row: row.minute, 1)
            # 对于5分钟间隔数据，使用minute//5作为时间槽
            df_stamp['minute'] = df_stamp['minute'] // 5
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.asarray(data_stamp, dtype=np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    """
    通用自定义数据集加载器
    适用于任意包含时间戳和数值特征的CSV/Excel文件
    """
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)

        # 支持CSV和Excel格式
        if local_fp.endswith('.csv'):
            df_raw = pd.read_csv(local_fp)
        elif local_fp.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(local_fp)
        else:
            raise ValueError(f"Unsupported file format: {local_fp}")

        '''
        df_raw.columns: ['date' or 'timestamp', ...(other features), target feature]
        '''
        # 尝试识别时间戳列
        time_col = None
        for col in ['timestamp', 'date', 'datetime', 'time']:
            if col in df_raw.columns:
                time_col = col
                break
        
        if time_col is None:
            # 如果没有时间戳列，添加一个自增序列
            df_raw.insert(0, 'date', pd.date_range(start='2020-01-01', periods=len(df_raw), freq='h'))
            time_col = 'date'
        
        # 获取所有特征列（除了时间戳）
        cols = [c for c in df_raw.columns if c != time_col]
        
        # 确保目标列存在
        if self.target not in cols:
            self.target = cols[-1]  # 默认使用最后一个列作为目标
            
        # 重新排列列顺序：时间 + 特征 + 目标
        cols.remove(self.target)
        df_raw = df_raw[[time_col] + cols + [self.target]]

        # 数据集划分（按时间顺序）
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 除了时间戳的所有列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        # 时间特征编码
        df_stamp = df_raw[[time_col]][border1:border2]
        df_stamp[time_col] = pd.to_datetime(df_stamp[time_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[time_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[time_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[time_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[time_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([time_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[time_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.asarray(data_stamp, dtype=np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
