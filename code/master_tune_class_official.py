import argparse
from typing import Optional
import copy
import torch
import torch.optim as optim

import numpy as np
import pandas as pd

from typing import Tuple, Union, List
from copy import deepcopy
from tqdm import tqdm
import bisect
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

import pickle
import os
import json
from itertools import product

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings

warnings.filterwarnings("ignore")


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
    """

    def __init__(
            self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None
    ):
        """
        初始化时间序列数据集

        参数
        ----------
        data : pd.DataFrame
            原始表格数据
        start :
            可索引的开始时间
        end :
            可索引的结束时间
        step_len : int
            时间序列步长
        fillna_type : str
            缺失值填充方式：'none', 'ffill', 'ffill+bfill'
        dtype :
            数据类型
        flt_data : pd.Series
            用于过滤数据的布尔列
        """
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
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(np.bool)
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
        """
        将索引映射转换为数组
        """
        dtype = np.int32
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        """
        根据过滤数据更新索引映射
        """
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        """
        获取数据集索引
        """
        return self.data_index[self.start_idx: self.end_idx]

    def config(self, **kwargs):
        """
        配置采样器参数
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        构建索引映射
        """
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        """
        检查数据集是否为空
        """
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        获取时间序列索引
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        获取行列索引
        """
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        获取数据
        """
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        data = self.data_arr[indices]
        if isinstance(idx, mtit):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        """
        获取数据集长度
        """
        return len(self.idx_map)


class DailyBatchSamplerRandom(Sampler):
    """
    按时间分组的批次采样器
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

        index_df = self.data_source.get_index()
        self.daily_count = pd.Series(index=index_df).groupby(level="time_id").size().values

        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

        print(f"数据集总样本数: {len(data_source)}")
        print(f"时间分组数量: {len(self.daily_count)}")
        print(
            f"每个时间组的样本数: 最小 {min(self.daily_count)}, 最大 {max(self.daily_count)}, 平均 {np.mean(self.daily_count):.2f}")

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
    序列模型基类
    """

    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path='../models/',
                 save_prefix=''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False
        self.model = None
        self.train_optimizer = None

        self.loss_fn = nn.CrossEntropyLoss()

        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        """
        初始化模型和优化器
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def compute_metrics(self, preds, labels):
        """
        计算分类指标 (准确率, 精确率, 召回率, F1分数)

        参数
        ----------
        preds : np.ndarray
            模型预测的类别 (0, 1, or 2)
        labels : np.ndarray
            真实的类别标签

        返回
        -------
        metrics : dict
            包含 accuracy, precision, recall, f1 的字典
        """
        mask = ~np.isnan(labels)
        preds = preds[mask]
        labels = labels[mask]

        if len(labels) < 2:
            return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return metrics

    def _init_data_loader(self, data, shuffle=False):
        """
        初始化 DataLoader
        """
        batch_sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, batch_sampler=batch_sampler)
        return data_loader

    def fit(self, dl_train, dl_valid, patience=10, min_delta=0.001):
        """
        训练模型
        """
        train_loader = self._init_data_loader(dl_train, shuffle=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False)

        self.fitted = True
        best_param = None
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # %===== 新增(1): 初始化用于存储训练历史的列表 =====
        # 这个列表将记录每个epoch的训练和验证损失。
        # =================================================
        epoch_history = []

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print(f"Epoch {step}, train_loss {train_loss:.6f}, valid_loss {val_loss:.6f}")

            # %===== 新增(2): 记录当前epoch的损失 =====
            # 将当前epoch的编号、训练损失和验证损失存入历史记录列表。
            # =========================================
            epoch_history.append({'epoch': step, 'train_loss': train_loss, 'valid_loss': val_loss})

            if np.isnan(val_loss):
                print("验证损失为NaN，提前停止训练。")
                break

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_param = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"连续 {patience} 个epoch验证损失没有提升，提前停止。")
                break

        model_name = f"{self.save_prefix}_d{self.d_model}_t{self.t_nhead}_s{self.s_nhead}_do{self.T_dropout_rate}_lr{self.lr}"

        # %===== 新增(3): 保存训练历史到CSV文件 =====
        # 在训练结束后，将记录的损失历史转换为DataFrame并保存为CSV文件。
        # 文件名与模型名关联，方便查找。
        # ==========================================
        history_df = pd.DataFrame(epoch_history)
        history_path = os.path.join(self.save_path, f"{model_name}_loss_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"训练历史已保存至: {history_path}")

        if best_param is not None:
            # %===== 修改: 同时保存为 .pkl 和 .pt 格式 =====
            # 根据您的要求，我们将最佳模型的状态字典同时保存为两种格式的文件。
            # .pkl 是Python的通用序列化格式，.pt 是PyTorch的推荐格式。
            # ===============================================
            model_pkl_path = os.path.join(self.save_path, f"{model_name}.pkl")
            model_pt_path = os.path.join(self.save_path, f"{model_name}.pt")

            torch.save(best_param, model_pkl_path)
            torch.save(best_param, model_pt_path)
            print(f"模型已保存至: {model_pkl_path} 和 {model_pt_path}")

            self.model.load_state_dict(best_param)
        else:
            print("没有找到最佳模型，不保存。")
            self.fitted = False

    def train_epoch(self, data_loader):
        """
        训练一个 epoch
        """
        self.model.train()
        losses = []
        for batch_data in tqdm(data_loader, desc="训练中", leave=False):
            feature = batch_data[:, :, :-1].to(self.device)
            label = batch_data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label.long())

            if not torch.isnan(loss):
                losses.append(loss.item())
                self.train_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()

        return float(np.mean(losses)) if losses else np.nan

    def test_epoch(self, data_loader):
        """
        测试一个 epoch
        """
        self.model.eval()
        losses = []
        for batch_data in tqdm(data_loader, desc="验证中", leave=False):
            feature = batch_data[:, :, :-1].to(self.device)
            label = batch_data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label.long())
                if not torch.isnan(loss):
                    losses.append(loss.item())

        return float(np.mean(losses)) if losses else np.nan

    def predict(self, dl_test):
        """
        进行分类预测和评估

        参数
        ----------
        dl_test : TSDataSampler
            测试数据集

        返回
        -------
        predictions : pd.Series
            预测的类别序列
        all_labels : np.ndarray
            真实的类别标签序列
        metrics : dict
            包含 accuracy, precision, recall, f1 的字典
        """
        if not self.fitted:
            print("模型未被成功拟合，跳过预测。")
            return None, None, self.compute_metrics(np.array([]), np.array([]))

        test_loader = self._init_data_loader(dl_test, shuffle=False)

        all_preds = []
        all_labels = []
        all_indices = []

        self.model.eval()
        for batch_indices in test_loader.batch_sampler:
            batch_data = dl_test[batch_indices]
            batch_index = dl_test.get_index()[batch_indices]

            batch_data = torch.from_numpy(batch_data).float()

            feature = batch_data[:, :, :-1].to(self.device)
            label = batch_data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred_logits = self.model(feature)
                pred = torch.argmax(pred_logits, dim=1).detach().cpu().numpy()
                label = label.cpu().numpy()

            all_preds.append(pred)
            all_labels.append(label)
            all_indices.append(batch_index)

        if not all_preds:
            return None, None, self.compute_metrics(np.array([]), np.array([]))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_indices = pd.concat([pd.Series(0, index=idx) for idx in all_indices]).index

        full_index = dl_test.get_index()
        predictions = pd.Series(all_preds, index=full_index)

        metrics = self.compute_metrics(all_preds, all_labels)
        return predictions, all_labels, metrics


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """

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
    """
    空间注意力模块
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(nn.Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    """
    时间注意力模块
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(nn.Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TemporalAttention(nn.Module):
    """
    时间注意力聚合模块
    """

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)
        return output


class MASTER(nn.Module):
    """
    MASTER 模型
    """

    def __init__(self, d_feat=133, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 num_classes=3):
        super(MASTER, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class MASTERModel(SequenceModel):
    """
    MASTER 模型实现
    """

    def __init__(
            self, d_feat: int = 133, d_model: int = 256, t_nhead: int = 4, s_nhead: int = 2,
            T_dropout_rate=0.5, S_dropout_rate=0.5, num_classes=3, **kwargs,
    ):
        self.d_model = d_model
        self.d_feat = d_feat
        self.num_classes = num_classes
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead

        super(MASTERModel, self).__init__(**kwargs)
        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            num_classes=self.num_classes)
        super(MASTERModel, self).init_model()


def main(args):
    """
    主函数，执行数据预处理、模型训练和测试
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet')
    data = pd.read_parquet(data_path)

    labels = [i for i in data.columns if i.startswith('label_')]
    labels_to_drop = [i for i in labels if i not in ['label_vwap_5m_ud']]
    data = data.drop(columns=labels_to_drop, axis=1)
    data.rename(columns={'label_vwap_5m_ud': 'label'}, inplace=True)

    data['datetime'] = pd.to_datetime(data['datetime'])

    date_select_begin = '2024-09-01'
    data = data[data['datetime'] >= date_select_begin].copy()
    print(f"目前datetime的最小值: {data['datetime'].min()}")

    print("正在增加时间特征...")
    data['minute_of_day'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute
    minutes_in_day = 24 * 60
    data['time_sin'] = np.sin(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data['time_cos'] = np.cos(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data = data.drop(columns=['minute_of_day'])
    print("时间特征增加完成。")

    valid_start_date = '2024-12-01'
    test_start_date = '2024-12-15'
    print(f"验证集开始日期: {valid_start_date}, 测试集开始日期: {test_start_date}")

    train_data = data[data['datetime'] < valid_start_date].copy()
    valid_data = data[(data['datetime'] >= valid_start_date) & (data['datetime'] < test_start_date)].copy()
    test_data = data[data['datetime'] >= test_start_date].copy()

    for df in [train_data, valid_data, test_data]:
        df['time_id'] = pd.factorize(df['datetime'])[0].astype(np.float32)

    for df in [train_data, valid_data, test_data]:
        df.drop(columns=['datetime'], inplace=True)
        label_col = df.pop('label')
        df.insert(len(df.columns), 'label', label_col)
        for col in df.columns:
            if col not in ['instrument', 'time_id']:
                df[col] = df[col].astype(np.float32)
        df.set_index(['time_id', 'instrument'], inplace=True)

    d_feat_calculated = len(train_data.columns) - 1
    print(f"动态计算出的特征维度 (d_feat): {d_feat_calculated}")
    args.d_feat = d_feat_calculated

    train_min_tid = train_data.index.get_level_values('time_id').min()
    train_max_tid = train_data.index.get_level_values('time_id').max()
    valid_min_tid = valid_data.index.get_level_values('time_id').min()
    valid_max_tid = valid_data.index.get_level_values('time_id').max()
    test_min_tid = test_data.index.get_level_values('time_id').min()
    test_max_tid = test_data.index.get_level_values('time_id').max()

    train_dataset = TSDataSampler(data=train_data, start=train_min_tid, end=train_max_tid, step_len=10,
                                  fillna_type='ffill+bfill')
    valid_dataset = TSDataSampler(data=valid_data, start=valid_min_tid, end=valid_max_tid, step_len=10,
                                  fillna_type='ffill+bfill')
    test_dataset = TSDataSampler(data=test_data, start=test_min_tid, end=test_max_tid, step_len=10,
                                 fillna_type='ffill+bfill')

    model = MASTERModel(
        d_feat=args.d_feat, d_model=args.d_model, t_nhead=args.t_nhead, s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout, S_dropout_rate=args.dropout,
        n_epochs=args.n_epoch, lr=args.lr, GPU=args.gpu, seed=args.seed,
        save_path=args.save_path, save_prefix=args.save_prefix,
        num_classes=3
    )

    model.fit(train_dataset, valid_dataset)

    predictions, labels, metrics = model.predict(test_dataset)
    print(f"测试集评估指标:")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_feat', type=int, default=133,
                        help='输入特征维度，此值会被动态计算覆盖')
    parser.add_argument('--d_model', type=int, nargs='+', default=[64, 128],
                        help='模型隐层维度，候选值列表')
    parser.add_argument('--t_nhead', type=int, nargs='+', default=[2],
                        help='时序注意力头数，候选值列表')
    parser.add_argument('--s_nhead', type=int, nargs='+', default=[1],
                        help='空间注意力头数，候选值列表')
    parser.add_argument('--dropout', type=float, nargs='+', default=[0.5, 0.6],
                        help='Dropout概率，候选值列表')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, nargs='+', default=[5e-5, 1e-4, 2e-4],
                        help='学习率')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_path = os.path.join(base_dir, 'models/MASTER_tuned_classification')
    parser.add_argument('--save_path', type=str, default=default_save_path,
                        help='模型保存路径')
    parser.add_argument('--save_prefix', type=str, default='master_clf',
                        help='模型文件名前缀')

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if not os.path.isabs(args.save_path):
        args.save_path = os.path.abspath(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"模型将保存到: {args.save_path}")

    param_grid = {
        'd_model': args.d_model,
        't_nhead': args.t_nhead,
        's_nhead': args.s_nhead,
        'dropout': args.dropout,
        'lr': args.lr
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_score = float('inf')
    best_params = {}

    # %===== 新增(4): 初始化用于存储所有运行结果的列表 =====
    # 这个列表将汇总每次参数组合的配置及其最终的评估指标。
    # ====================================================
    all_runs_metrics = []

    for params in param_combinations:
        current_args = argparse.Namespace(**vars(args))
        for key, value in params.items():
            setattr(current_args, key, value)
        print(f"\n测试参数组合: {params}")

        # %===== 新增: 检查模型是否已存在，若存在则跳过 =====
        # 根据当前参数组合构建预期的模型文件名。
        model_name_check = (
            f"{current_args.save_prefix}_d{current_args.d_model}"
            f"_t{current_args.t_nhead}_s{current_args.s_nhead}"
            f"_do{current_args.dropout}_lr{current_args.lr}"
        )
        model_path_check_pkl = os.path.join(current_args.save_path, f"{model_name_check}.pkl")
        model_path_check_pt = os.path.join(current_args.save_path, f"{model_name_check}.pt")

        # 检查 .pkl 或 .pt 文件是否存在
        if os.path.exists(model_path_check_pkl) or os.path.exists(model_path_check_pt):
            print(f"模型 '{model_name_check}' 已存在，跳过此参数组合的训练。")
            continue
        # ======================================================

        try:
            metrics = main(current_args)
            if metrics is None or np.isnan(metrics['accuracy']):
                print(f"参数组合 {params} 结果为 NaN，跳过。")
                continue
        except Exception as e:
            print(f"参数组合 {params} 运行失败，错误: {e}")
            continue

        # %===== 新增(5): 记录本次运行的参数和结果 =====
        # 将当前的参数配置和得到的评估指标合并到一个字典中，然后添加到总的记录列表里。
        # ============================================
        run_result = params.copy()
        run_result.update(metrics)
        all_runs_metrics.append(run_result)

        score = 1 - metrics['accuracy']
        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_params.update(metrics)
            print(f"找到新的最佳参数组合，准确率: {metrics['accuracy']:.6f}")
            print(f"当前最佳参数详情: {best_params}")

    print("\n======== 超参数搜索完成 ========")
    print("最佳参数组合:")
    if best_params:
        for k, v in best_params.items():
            print(f"{k}: {v}")

        # %===== 新增(6): 保存所有运行结果到CSV文件 =====
        # 将超参数搜索的完整记录保存到一个CSV文件中，便于后续分析和比较。
        # =============================================
        if all_runs_metrics:
            results_df = pd.DataFrame(all_runs_metrics)
            results_csv_path = os.path.join(args.save_path, f"{args.save_prefix}_all_runs_metrics.csv")
            results_df.to_csv(results_csv_path, index=False)
            print(f"所有运行的指标已保存至: {results_csv_path}")

        with open(os.path.join(args.save_path, f"{args.save_prefix}_best_params.pkl"), 'wb') as f:
            pickle.dump(best_params, f)
        print(f"最佳参数已保存到: {os.path.join(args.save_path, f'{args.save_prefix}_best_params.pkl')}")

        params_to_save = best_params.copy()
        for key, value in params_to_save.items():
            if isinstance(value, (np.float32, np.float64)):
                params_to_save[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                params_to_save[key] = int(value)

        with open(os.path.join(args.save_path, f"{args.save_prefix}_best_params.json"), 'w') as f:
            json.dump(params_to_save, f, indent=4)
        print(f"最佳参数 (JSON格式) 已保存到: {os.path.join(args.save_path, f'{args.save_prefix}_best_params.json')}")
    else:
        print("所有参数组合均未成功运行，未找到最佳参数。")
