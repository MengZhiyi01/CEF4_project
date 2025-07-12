import argparse
from typing import Optional, Tuple, Union, List
import copy
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import polars as pl
from copy import deepcopy
from tqdm import tqdm
import bisect
from torch.utils.data import DataLoader, Sampler
from torch import nn
import math
import pickle
import os
import json
import warnings

warnings.filterwarnings("ignore")


# %=================================================================
# 任务1: 环境准备与类定义
# 这一步包含了所有必要的库导入和类定义。
# 这些定义已更新，与训练脚本(master5.py)完全一致，以确保模型可以被正确加载。
# %=================================================================

def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    对 DataFrame 按索引或列进行排序（如果未排序）
    """
    idx = df.index if axis == 0 else df.columns
    if not idx.is_monotonic_increasing:
        return df.sort_index(axis=axis)
    else:
        return df


def np_ffill(arr: np.array):
    """
    对 NumPy 数组进行前向填充
    """
    mask = np.isnan(arr.astype(float))
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


class TSDataSampler:
    """
    时间序列数据采样器，类似 torch.utils.data.Dataset
    (版本与 master5.py 兼容)
    """

    def __init__(
            self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None
    ):
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        self.data = lazy_sort_index(data)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)
        self.data_arr = np.append(
            self.data_arr, np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype), axis=0
        )
        self.nan_idx = -1

        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)

        self.start_idx = self.data_index.get_level_values(0).searchsorted(start, side='left')
        self.end_idx = self.data_index.get_level_values(0).searchsorted(end, side='right')

        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)
        del self.data

    @staticmethod
    def idx_map2arr(idx_map):
        dtype = np.int32
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)
        max_idx = max(idx_map.keys()) if idx_map else -1
        arr_map = [idx_map.get(i, no_existing_idx) for i in range(max_idx + 1)]
        return np.array(arr_map, dtype=dtype)

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        return self.data_index[self.start_idx: self.end_idx]

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        idx_df = lazy_sort_index(idx_df, axis=1)
        idx_map = {int(real_idx): (i, j) for i, (_, row) in enumerate(idx_df.iterrows()) for j, real_idx in
                   enumerate(row) if not np.isnan(real_idx)}
        return idx_df, idx_map

    def _get_indices(self, row: int, col: int) -> np.array:
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])
        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        return indices

    def _get_row_col(self, idx) -> Tuple[int, int]:
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                return self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
            return i, j
        raise NotImplementedError("This type of input is not supported")

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int], np.ndarray]):
        if isinstance(idx, (list, np.ndarray)):
            indices = np.concatenate([self._get_indices(*self._get_row_col(i)) for i in idx])
        else:
            indices = self._get_indices(*self._get_row_col(idx))
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        data = self.data_arr[indices]
        if isinstance(idx, (list, np.ndarray)):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return self.end_idx - self.start_idx


class DailyBatchSamplerRandom(Sampler):
    """
    按时间分组的批次采样器 (版本与 master5.py 兼容)
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        index_df = self.data_source.get_index()
        self.daily_count = pd.Series(index=index_df).groupby(level="time_id").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        print(f"数据采样器初始化：找到 {len(self.daily_count)} 个时间组。")

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.daily_count)


class SequenceModel:
    """
    序列模型基类 (版本与 master5.py 兼容)
    """

    def __init__(self, n_epochs=0, lr=0.0, GPU=None, seed=None, **kwargs):
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.model = None
        self.fitted = False

    def init_model(self, **kwargs):
        raise NotImplementedError("init_model must be implemented by subclass")

    def _init_data_loader(self, data, shuffle=False):
        batch_sampler = DailyBatchSamplerRandom(data, shuffle)
        return DataLoader(data, batch_sampler=batch_sampler)

    def compute_metrics(self, preds, labels):
        mask = ~np.isnan(labels)
        preds, labels = preds[mask], labels[mask]
        if len(labels) < 2:
            return np.nan, np.nan, np.nan
        mse = np.mean((preds - labels) ** 2)
        denominator_r2 = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - np.sum((labels - preds) ** 2) / (denominator_r2 + 1e-8)
        ic = np.corrcoef(preds, labels)[0, 1] if denominator_r2 > 1e-8 else np.nan
        return mse, r2, ic

    def predict(self, dl_test, label_scaling_factor=1.0):
        if not self.fitted:
            print("模型未被加载或拟合，跳过预测。")
            return None, None, np.nan, np.nan, np.nan

        test_loader = self._init_data_loader(dl_test, shuffle=False)
        all_preds, all_labels = [], []
        self.model.eval()

        for batch_indices in tqdm(test_loader.batch_sampler, desc="预测中", leave=False):
            # 直接使用索引从 TSDataSampler 获取数据以确保对齐
            batch_data = dl_test[batch_indices]
            batch_data = torch.from_numpy(batch_data).float()

            feature = batch_data[:, :, :-1].to(self.device)
            label = batch_data[:, -1, -1].cpu().numpy()

            with torch.no_grad():
                pred = self.model(feature).detach().cpu().numpy()

            all_preds.append(pred)
            all_labels.append(label)

        if not all_preds:
            return None, None, np.nan, np.nan, np.nan

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        all_preds_rescaled = all_preds / label_scaling_factor
        all_labels_rescaled = all_labels / label_scaling_factor

        full_index = dl_test.get_index()
        predictions = pd.Series(all_preds_rescaled, index=full_index)

        mse, r2, ic = self.compute_metrics(all_preds_rescaled, all_labels_rescaled)
        return predictions, all_labels_rescaled, mse, r2, ic


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.temperature = math.sqrt(d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=dropout),
                                 nn.Linear(d_model, d_model), nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.norm1(x)
        q, k, v = self.qtrans(x).transpose(0, 1), self.ktrans(x).transpose(0, 1), self.vtrans(x).transpose(0, 1)
        dim = self.d_model // self.nhead
        att_output = []
        for i in range(self.nhead):
            start, end = i * dim, (i + 1) * dim
            qh, kh, vh = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.cat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=dropout),
                                 nn.Linear(d_model, d_model), nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.norm1(x)
        q, k, v = self.qtrans(x), self.ktrans(x), self.vtrans(x)
        dim = self.d_model // self.nhead
        att_output = []
        for i in range(self.nhead):
            start, end = i * dim, (i + 1) * dim
            qh, kh, vh = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.cat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTER(nn.Module):
    def __init__(self, d_feat=133, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5):
        super(MASTER, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MASTERModel(SequenceModel):
    def __init__(self, d_feat: int = 133, d_model: int = 256, t_nhead: int = 4, s_nhead: int = 2,
                 T_dropout_rate=0.5, S_dropout_rate=0.5, **kwargs):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate)
        self.model.to(self.device)


# %=================================================================
# 任务2: 加载最优模型
# %=================================================================

# --- 配置模型和数据路径 ---
# 这个路径应该指向你运行 master5.py 时模型保存的目录
# 例如：'models/MASTER_tuned'
MODEL_SAVE_DIR = '../experiments/models/MASTER_tuned'
# 模型文件名的前缀，也需要和 master5.py 中一致
# 例如：'master'
MODEL_PREFIX = 'master'

base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_dir = os.path.join(base_dir, MODEL_SAVE_DIR)

# --- 加载最优参数 ---
params_path = os.path.join(model_dir, f"{MODEL_PREFIX}_best_params.json")
print(f"正在从 {params_path} 加载最优参数...")
try:
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print("最优参数加载成功:")
    print(best_params)
except FileNotFoundError:
    print(f"错误: 找不到参数文件 {params_path}")
    print("请确保 MODEL_SAVE_DIR 和 MODEL_PREFIX 变量设置正确，并且训练已完成。")
    exit()

# --- 实例化模型 ---
# 我们需要从参数文件中提取模型结构相关的参数
model_params = {
    'd_model': best_params['d_model'],
    't_nhead': best_params['t_nhead'],
    's_nhead': best_params['s_nhead'],
    'dropout': best_params['dropout'],
    'lr': best_params['lr']
}

# d_feat 将在数据处理后动态确定
d_feat_placeholder = 133  # 占位符

# --- 构造模型文件名 (与 master5.py 匹配) ---
model_filename = (
    f"{MODEL_PREFIX}_"
    f"dmodel{model_params['d_model']}_"
    f"tn{model_params['t_nhead']}_"
    f"sn{model_params['s_nhead']}_"
    f"do{model_params['dropout']}_"
    f"lr{model_params['lr']}.pkl"
)
model_path = os.path.join(model_dir, model_filename)


# %=================================================================
# 任务3: 数据处理与预测
# %=================================================================

def process_data_for_prediction(data_path, test_start_date, label_scaling_factor=1000.0):
    """
    加载并处理数据以用于预测。
    这个函数包含了清理、特征工程、缩放和数据集创建的完整流程。
    (逻辑与 master5.py 一致)
    """
    print("\n--- 开始数据处理 ---")
    # 加载数据
    try:
        data = pl.read_parquet(data_path).to_pandas()
    except Exception as e:
        print(f"错误：无法读取数据文件 {data_path}。错误信息: {e}")
        exit()

    # 1. 清理无穷大值和NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # 2. 筛选和重命名label
    labels = [i for i in data.columns if i.startswith('label_')]
    labels_to_drop = [i for i in labels if i not in ['label_vwap_5m']]
    data = data.drop(labels_to_drop, axis=1)
    data.rename(columns={'label_vwap_5m': 'label'}, inplace=True)

    # 3. 确保datetime列是datetime类型并筛选测试数据
    data['datetime'] = pd.to_datetime(data['datetime'])
    test_data = data[data['datetime'] >= test_start_date].copy()
    if test_data.empty:
        print(f"错误：在起始日期 {test_start_date} 之后没有找到数据。请检查日期或数据文件。")
        return None, -1
    print(f"已筛选出测试数据，开始日期: {test_data['datetime'].min()}")

    # 4. 增加时间特征 (与训练时一致)
    test_data['minute_of_day'] = test_data['datetime'].dt.hour * 60 + test_data['datetime'].dt.minute
    minutes_in_day = 24 * 60
    test_data['time_sin'] = np.sin(2 * np.pi * test_data['minute_of_day'] / minutes_in_day)
    test_data['time_cos'] = np.cos(2 * np.pi * test_data['minute_of_day'] / minutes_in_day)
    test_data = test_data.drop(columns=['minute_of_day'])

    # 5. 对label进行缩放 (与训练时一致)
    test_data['label'] = test_data['label'] * label_scaling_factor

    # 6. 创建time_id并整理列顺序
    test_data['time_id'] = pd.factorize(test_data['datetime'])[0].astype(np.float32)
    test_data.drop(columns=['datetime'], inplace=True)
    label_col = test_data.pop('label')
    test_data.insert(len(test_data.columns), 'label', label_col)

    # 7. 转换数据类型并设置索引
    for col in test_data.columns:
        if col not in ['instrument', 'time_id']:
            test_data[col] = test_data[col].astype(np.float32)
    test_data.set_index(['time_id', 'instrument'], inplace=True)

    # 8. 动态计算特征维度
    d_feat_calculated = len(test_data.columns) - 1
    print(f"测试数据计算出的特征维度 (d_feat): {d_feat_calculated}")

    # 9. 创建TSDataSampler实例
    test_min_tid = test_data.index.get_level_values('time_id').min()
    test_max_tid = test_data.index.get_level_values('time_id').max()

    test_dataset = TSDataSampler(data=test_data, start=test_min_tid, end=test_max_tid + 1, step_len=10,
                                 fillna_type='ffill+bfill')

    print("--- 数据处理完成 ---")
    return test_dataset, d_feat_calculated


def calculate_and_display_per_asset_metrics(predictions: pd.Series, labels: np.ndarray):
    """
    计算并展示每个资产的R2和IC，以及它们的平均值。
    """
    if predictions is None or labels is None or predictions.empty:
        print("无法计算每个资产的指标，因为预测或标签为空。")
        return

    results_df = pd.DataFrame({
        'prediction': predictions,
        'label': pd.Series(labels, index=predictions.index)
    })
    results_df.dropna(inplace=True)
    results_df.reset_index(inplace=True)

    asset_groups = results_df.groupby('instrument')
    asset_metrics = []

    for asset, group in asset_groups:
        if len(group) < 20: # 样本太少则跳过
            continue

        # 计算 R2
        y_true = group['label']
        y_pred = group['prediction']
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-8)

        # 计算 IC (Pearson correlation)
        ic = np.corrcoef(y_pred, y_true)[0, 1]

        if not np.isnan(r2) and not np.isnan(ic):
             asset_metrics.append({'asset': asset, 'r2': r2, 'ic': ic, 'sample_count': len(group)})

    if not asset_metrics:
        print("没有足够的资产数据来计算指标。")
        return

    metrics_df = pd.DataFrame(asset_metrics)
    avg_r2 = metrics_df['r2'].mean()
    avg_ic = metrics_df['ic'].mean()

    print("\n--- 每个资产的评估指标 ---")
    print(metrics_df.to_string(index=False))

    print("\n--- 平均指标 (跨资产) ---")
    print(f"  - 平均 R2: {avg_r2:.6f}")
    print(f"  - 平均 IC:  {avg_ic:.6f}")


# %=================================================================
# 任务4: 执行预测并展示结果
# %=================================================================

# --- 定义新的测试时间段和数据路径 ---
# 你可以修改这里的日期来选择不同的时间段进行预测
NEW_TEST_START_DATE = '2024-12-30'
# 确保这个路径指向你的数据文件
RAW_DATA_PATH = os.path.join(base_dir, '../data/output/final_filtered_data_1min_0708v3.parquet')
# 必须与训练时使用的缩放因子一致
LABEL_SCALING_FACTOR = 1000.0

# --- 处理新数据 ---
test_dataset, d_feat_actual = process_data_for_prediction(
    data_path=RAW_DATA_PATH,
    test_start_date=NEW_TEST_START_DATE,
    label_scaling_factor=LABEL_SCALING_FACTOR
)

if test_dataset is None:
    exit()

# --- 使用正确的d_feat重新实例化并加载模型 ---
print(f"\n使用正确的特征维度 ({d_feat_actual}) 加载模型...")
model = MASTERModel(
    d_feat=d_feat_actual,
    d_model=model_params['d_model'],
    t_nhead=model_params['t_nhead'],
    s_nhead=model_params['s_nhead'],
    T_dropout_rate=model_params['dropout'],
    S_dropout_rate=model_params['dropout']
)

try:
    model.model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.fitted = True
    print(f"模型权重从 {model_path} 加载成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 {model_path}")
    print("请确认模型文件是否存在，以及文件名是否与参数匹配。")
    exit()

# --- 执行预测 ---
predictions, labels, mse, r2, ic = model.predict(
    test_dataset,
    label_scaling_factor=LABEL_SCALING_FACTOR
)

# --- 展示结果 ---
if predictions is not None:
    print("\n--- 预测完成 ---")
    print(f"\n测试集 {NEW_TEST_START_DATE} 之后的数据总体评估结果:")
    print(f"  - 整体 MSE: {mse:.6f}")
    print(f"  - 整体 R2:  {r2:.6f}")
    print(f"  - 整体 IC:  {ic:.6f}")

    print("\n预测结果 (前5条):")
    print(predictions.head())

    # --- 计算并展示每个资产的指标 ---
    calculate_and_display_per_asset_metrics(predictions, labels)

else:
    print("\n预测失败。")