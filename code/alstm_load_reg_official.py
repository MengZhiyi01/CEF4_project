import argparse
from typing import Optional, Tuple, Union, List
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import json
import pickle
import warnings

warnings.filterwarnings("ignore")


# %=================================================================
# 任务1: 环境准备与类定义
# %=================================================================

class ALSTMModel(nn.Module):
    """
    ALSTM模型的核心结构。
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_instruments=0, embedding_dim=4):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_instruments = n_instruments
        self.embedding_dim = embedding_dim

        self.instrument_embedding = nn.Embedding(num_embeddings=self.n_instruments, embedding_dim=self.embedding_dim)
        self.fc_in = nn.Linear(in_features=self.d_feat + self.embedding_dim, out_features=self.hidden_size)
        self.act = nn.Tanh()
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.att_net = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size / 2)),
            nn.Dropout(self.dropout),
            nn.Tanh(),
            nn.Linear(in_features=int(self.hidden_size / 2), out_features=1, bias=False),
            nn.Softmax(dim=1)
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=1)

    def forward(self, x_continuous, x_instrument):
        instrument_embed = self.instrument_embedding(x_instrument)
        x = torch.cat([x_continuous, instrument_embed], dim=2)
        x = self.act(self.fc_in(x))
        rnn_out, _ = self.rnn(x)
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out.squeeze(-1)


class TimeSeriesDataset(Dataset):
    """
    自定义的PyTorch数据集类，用于处理时序数据。
    """

    def __init__(self, df: pd.DataFrame, step_len=10):
        self.step_len = step_len
        df_sorted = df.sort_values(by=['instrument_idx', 'datetime']).reset_index(drop=True)

        feature_cols = [col for col in df_sorted.columns if
                        col not in ['instrument_idx', 'label', 'datetime', 'instrument']]
        self.instrument_indices = df_sorted['instrument_idx'].values.astype(np.int64)
        self.labels = df_sorted['label'].values.astype(np.float32)
        self.features = df_sorted[feature_cols].values.astype(np.float32)
        self.original_df_sorted = df_sorted

        self.samples = []
        boundaries = np.where(df_sorted['instrument_idx'].diff() != 0)[0]
        group_starts = np.concatenate(([0], boundaries))
        group_ends = np.concatenate((boundaries, [len(df_sorted)]))

        for start, end in zip(group_starts, group_ends):
            group_len = end - start
            if group_len >= step_len:
                for i in range(step_len - 1, group_len):
                    self.samples.append(start + i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        end_idx = self.samples[idx]
        start_idx = end_idx - self.step_len + 1
        features_sample = self.features[start_idx:end_idx + 1]
        instrument_sample = self.instrument_indices[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        return torch.tensor(features_sample, dtype=torch.float), torch.tensor(instrument_sample,
                                                                              dtype=torch.long), torch.tensor(label,
                                                                                                              dtype=torch.float)


class ALSTMWrapper:
    """
    ALSTM模型的封装器。
    """

    def __init__(self, d_feat, hidden_size, num_layers, dropout, n_instruments, embedding_dim, GPU=0, seed=None,
                 **kwargs):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_instruments = n_instruments
        self.embedding_dim = embedding_dim

        if torch.backends.mps.is_available():
            self.device = torch.device(f"mps:{GPU}" if GPU is not None else "mps")
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{GPU}" if GPU is not None else "cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.fitted = False
        self.model = ALSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_instruments=self.n_instruments,
            embedding_dim=self.embedding_dim
        ).to(self.device)

    # %==================== 修改开始 ====================%
    # 中文注释：重写 load_model 函数以手动处理权重尺寸不匹配的问题。
    def load_model(self, model_path):
        """
        加载已保存的模型权重。
        如果当前模型的嵌入层尺寸大于checkpoint中的尺寸，则手动加载权重。
        """
        if not os.path.exists(model_path):
            print(f"错误：找不到模型文件 {model_path}")
            return

        # 加载保存在文件中的 state_dict (checkpoint)
        checkpoint = torch.load(model_path, map_location=self.device)
        # 获取当前新创建模型的 state_dict
        current_model_dict = self.model.state_dict()

        # 提取旧的嵌入层权重和当前模型的嵌入层权重
        pretrained_embedding_weight = checkpoint.get('instrument_embedding.weight')
        current_embedding_weight = current_model_dict.get('instrument_embedding.weight')

        # 检查嵌入层是否存在且尺寸不匹配
        if pretrained_embedding_weight is not None and pretrained_embedding_weight.shape != current_embedding_weight.shape:
            print("检测到嵌入层权重尺寸不匹配 (这是预期的)。正在手动加载权重...")

            # 1. 从checkpoint中筛选出所有尺寸匹配的层
            #    (除了instrument_embedding.weight，其他所有层都应该匹配)
            pretrained_dict = {
                k: v for k, v in checkpoint.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }

            # 2. 将这些匹配的权重加载到当前模型的 state_dict 中
            current_model_dict.update(pretrained_dict)

            # 3. 手动将旧的、较小的嵌入权重复制到新的、较大的嵌入矩阵的前N行
            n_pretrained_rows = pretrained_embedding_weight.shape[0]
            print(f"将 {n_pretrained_rows} 个已训练的资产嵌入复制到新模型中...")
            current_model_dict['instrument_embedding.weight'][:n_pretrained_rows, :] = pretrained_embedding_weight

            # 4. 将我们手动构建好的、完整的 state_dict 加载到模型中
            self.model.load_state_dict(current_model_dict)
            print("权重加载成功。新资产(unknown)的嵌入将使用随机初始化权重。")

        else:
            # 如果尺寸完全匹配（例如，你用修改后的tune脚本重新训练了模型），则直接加载
            print("模型权重尺寸匹配，直接加载...")
            self.model.load_state_dict(checkpoint)

        self.fitted = True
        print(f"模型已从 {model_path} 成功加载并准备就绪。")

    # %==================== 修改结束 ====================%

    def compute_metrics(self, preds, labels):
        mask = ~np.isnan(labels)
        preds, labels = preds[mask], labels[mask]
        if len(labels) < 2: return np.nan, np.nan, np.nan
        mse = np.mean((preds - labels) ** 2)
        denominator_r2 = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - np.sum((labels - preds) ** 2) / (denominator_r2 + 1e-8)
        ic = np.corrcoef(preds, labels)[0, 1] if len(labels) >= 2 else np.nan
        return mse, r2, ic

    def compute_per_instrument_metrics(self, predictions, labels, instrument_indices, instrument_map):
        """计算每个资产的IC和R2值并返回平均值"""
        ic_per_instrument = {}
        r2_per_instrument = {}
        unique_instruments = np.unique(instrument_indices)

        for inst_idx in unique_instruments:
            mask = instrument_indices == inst_idx
            if np.sum(mask) < 2:
                continue

            inst_preds = predictions[mask]
            inst_labels = labels[mask]
            mask_valid = ~np.isnan(inst_labels)

            if np.sum(mask_valid) < 2:
                continue

            valid_preds = inst_preds[mask_valid]
            valid_labels = inst_labels[mask_valid]

            ic = np.corrcoef(valid_preds, valid_labels)[0, 1]

            denominator_r2 = np.sum((valid_labels - np.mean(valid_labels)) ** 2)
            if denominator_r2 < 1e-8:
                r2 = np.nan
            else:
                r2 = 1 - np.sum((valid_labels - valid_preds) ** 2) / denominator_r2

            inst_name = instrument_map[inst_idx] if inst_idx < len(instrument_map) else f"unknown_{inst_idx}"
            ic_per_instrument[inst_name] = ic
            r2_per_instrument[inst_name] = r2

        mean_ic = np.nanmean(list(ic_per_instrument.values())) if ic_per_instrument else np.nan
        mean_r2 = np.nanmean(list(r2_per_instrument.values())) if r2_per_instrument else np.nan

        return ic_per_instrument, mean_ic, r2_per_instrument, mean_r2

    def predict(self, test_dataset, batch_size, instrument_map, label_scaling_factor=1.0):
        if not self.fitted:
            return None, None, np.nan, np.nan, np.nan, {}, np.nan, {}, np.nan

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_preds, all_labels, all_instrument_indices = [], [], []

        self.model.eval()
        with torch.no_grad():
            for x_continuous, x_instrument, y_batch in tqdm(test_loader, desc="预测中"):
                x_continuous = x_continuous.to(self.device)
                x_instrument = x_instrument.to(self.device)
                pred = self.model(x_continuous, x_instrument).cpu().numpy()
                all_preds.append(pred)
                all_labels.append(y_batch.numpy())
                all_instrument_indices.append(x_instrument[:, -1].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_instrument_indices = np.concatenate(all_instrument_indices)

        all_preds_rescaled = all_preds / label_scaling_factor
        all_labels_rescaled = all_labels / label_scaling_factor

        mse, r2, ic = self.compute_metrics(all_preds_rescaled, all_labels_rescaled)

        ic_per_instrument, mean_ic, r2_per_instrument, mean_r2 = self.compute_per_instrument_metrics(
            all_preds_rescaled, all_labels_rescaled, all_instrument_indices, instrument_map
        )

        prediction_rows = test_dataset.original_df_sorted.iloc[test_dataset.samples]
        datetime_index = prediction_rows['datetime']
        instrument_idx_index = prediction_rows['instrument_idx']
        instrument_name_index = instrument_idx_index.map(
            lambda x: instrument_map[x] if x >= 0 and x < len(instrument_map) else 'unknown')
        new_multi_index = pd.MultiIndex.from_arrays([datetime_index, instrument_name_index],
                                                    names=['datetime', 'instrument'])
        predictions = pd.Series(all_preds_rescaled, index=new_multi_index)

        return predictions, all_labels_rescaled, mse, r2, ic, ic_per_instrument, mean_ic, r2_per_instrument, mean_r2


# ... 文件的其余部分 (process_data_for_prediction 和 主执行流程) 保持不变 ...
# 我将它们也附在下面以保证文件的完整性

# %=================================================================
# 任务2: 数据处理函数
# %=================================================================
def process_data_for_prediction(data_path, test_start_date, label_scaling_factor=1000.0, instrument_map_path=None):
    print("--- 开始数据处理 ---")
    data = pd.read_parquet(data_path)
    # ... (省略未改变的代码)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    labels = [i for i in data.columns if i.startswith('label_')]
    data = data.drop([c for c in labels if c != 'label_vwap_5m'], axis=1)
    data.rename(columns={'label_vwap_5m': 'label'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['minute_of_day'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute
    minutes_in_day = 24 * 60
    data['time_sin'] = np.sin(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data['time_cos'] = np.cos(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data.drop(columns=['minute_of_day'], inplace=True)
    if instrument_map_path and os.path.exists(instrument_map_path):
        with open(instrument_map_path, 'rb') as f:
            instrument_map = pickle.load(f)
        instrument_to_idx = {name: idx for idx, name in enumerate(instrument_map)}
        n_instruments_train = len(instrument_map)
        data['instrument_idx'] = data['instrument'].map(instrument_to_idx).fillna(n_instruments_train).astype(int)
        n_instruments_for_model = n_instruments_train + 1
        print(f"成功加载训练时的instrument_map，共 {n_instruments_train} 个资产。")
        print(f"测试集中新出现的资产将被映射到索引 {n_instruments_train}。")
    else:
        print("错误：找不到 instrument_map 文件。无法进行预测，因为无法对齐资产。")
        raise FileNotFoundError(f"Instrument map file not found at {instrument_map_path}")
    data['label'] = data['label'] * label_scaling_factor
    test_data = data[data['datetime'] >= test_start_date].copy()
    print(f"已筛选出测试数据，开始日期: {test_data['datetime'].min()}")
    feature_cols = [col for col in data.columns if col not in ['datetime', 'instrument', 'label', 'instrument_idx']]
    final_cols_order = ['datetime', 'instrument', 'instrument_idx'] + feature_cols + ['label']
    test_data = test_data[final_cols_order]
    d_feat = len(feature_cols)
    print("--- 数据处理完成 ---")
    return test_data, n_instruments_for_model, d_feat, instrument_map


# %=================================================================
# 任务3: 主执行流程
# %=================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    parser = argparse.ArgumentParser()
    # ... (省略未改变的参数定义)
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_filtered_data_1min_0708v3.parquet'))
    parser.add_argument('--test_start_date', type=str, default='2024-12-15', help='测试集开始日期')
    parser.add_argument('--label_scale', type=float, default=1000.0)
    parser.add_argument('--step_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=os.path.join(base_dir, 'models/ALSTM_tuned/'))
    parser.add_argument('--instrument_map_path', type=str,
                        default=os.path.join(base_dir, 'models/ALSTM_tuned/instrument_map.pkl'),
                        help='Path to saved instrument map from training')
    args = parser.parse_args()
    BEST_PARAMS_PATH = os.path.join(args.save_path, "best_params_results.json")
    print(f"正在从 {BEST_PARAMS_PATH} 加载最优参数...")
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError(f"找不到最优参数文件: {BEST_PARAMS_PATH}，请先运行训练脚本。")
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    print("最优参数加载成功:", best_params)
    model_prefix = (
        f"alstm_hs{best_params['hidden_size']}_nl{best_params['num_layers']}_"
        f"do{best_params['dropout']}_ed{best_params['embedding_dim']}_lr{best_params['lr']}_bs{best_params['batch_size']}"
    )
    MODEL_PATH = os.path.join(args.save_path, f"{model_prefix}.pkl")
    test_data, n_instruments_for_model, d_feat, instrument_map = process_data_for_prediction(
        data_path=args.data_path,
        test_start_date=args.test_start_date,
        label_scaling_factor=args.label_scale,
        instrument_map_path=args.instrument_map_path
    )
    model = ALSTMWrapper(
        d_feat=d_feat,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        embedding_dim=best_params['embedding_dim'],
        n_instruments=n_instruments_for_model,
        GPU=args.gpu,
        seed=args.seed
    )
    model.load_model(MODEL_PATH)
    test_dataset = TimeSeriesDataset(test_data, step_len=args.step_len)
    predictions, _, mse, r2, ic, ic_per_instrument, mean_ic, r2_per_instrument, mean_r2 = model.predict(
        test_dataset,
        batch_size=args.batch_size,
        instrument_map=instrument_map,
        label_scaling_factor=args.label_scale
    )
    if predictions is not None:
        print("\n--- 最终预测结果 ---")
        # ... (省略未改变的打印部分)
        print(f"测试集 {args.test_start_date} 之后的数据评估结果:")
        print(f"  - MSE: {mse:.6f}")
        print(f"  - R2 (Overall):  {r2:.6f}")
        print(f"  - 平均 R2 (每个资产): {mean_r2:.6f}")
        print(f"  - 总体 IC:  {ic:.6f}")
        print(f"  - 平均 IC (每个资产): {mean_ic:.6f}")
        print("\n每个资产的R2值:")
        for inst, r2_val in sorted(r2_per_instrument.items()):
            if not np.isnan(r2_val):
                print(f"  - {inst}: {r2_val:.6f}")
        print("\n每个资产的IC值:")
        for inst, ic_val in sorted(ic_per_instrument.items()):
            if not np.isnan(ic_val):
                print(f"  - {inst}: {ic_val:.6f}")
        print("\n预测结果预览 (前10条):")
        print(predictions.head(10))
    else:
        print("\n预测失败。")