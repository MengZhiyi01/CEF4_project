import argparse
from typing import Optional, Tuple, Union, List
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import json
from itertools import product
import warnings
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


# %=================================================================
# 任务1: 模型定义与封装
# %=================================================================

class ALSTMModel(nn.Module):
    """
    ALSTM模型的核心结构。
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_instruments=0, embedding_dim=4,
                 n_classes=3):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_instruments = n_instruments
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

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
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=self.n_classes)

    def forward(self, x_continuous, x_instrument):
        instrument_embed = self.instrument_embedding(x_instrument)
        x = torch.cat([x_continuous, instrument_embed], dim=2)
        x = self.act(self.fc_in(x))
        rnn_out, _ = self.rnn(x)
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out


class TimeSeriesDataset(Dataset):
    """
    自定义的PyTorch数据集类，用于处理时序数据。
    """

    def __init__(self, df: pd.DataFrame, target_label_col: str, step_len=10):
        self.step_len = step_len
        df_sorted = df.sort_values(by=['instrument_idx', 'datetime']).reset_index(drop=True)

        feature_cols = [col for col in df_sorted.columns if not (
                col.startswith('label') or col in ['instrument_idx', 'datetime', 'instrument']
        )]
        self.instrument_indices = df_sorted['instrument_idx'].values.astype(np.int64)

        if target_label_col not in df_sorted.columns:
            raise ValueError(f"错误: 目标标签列 '{target_label_col}' 不在数据中！")
        self.labels = df_sorted[target_label_col].values.astype(np.int64)

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
        # 保持您原始代码的逻辑，每个时间步都取instrument_idx
        instrument_sample = self.instrument_indices[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        return torch.tensor(features_sample, dtype=torch.float), torch.tensor(instrument_sample,
                                                                              dtype=torch.long), torch.tensor(label,
                                                                                                              dtype=torch.long)


class ALSTMWrapper:
    """
    ALSTM模型的封装器。
    """

    def __init__(self, d_feat, hidden_size, num_layers, dropout, n_epochs, lr, n_instruments, embedding_dim,
                 n_classes=3, GPU=0,
                 seed=None, save_path='models/', save_prefix='alstm'):
        self.d_feat, self.hidden_size, self.num_layers, self.dropout, self.n_epochs, self.lr, self.n_instruments, self.embedding_dim, self.save_path, self.save_prefix = d_feat, hidden_size, num_layers, dropout, n_epochs, lr, n_instruments, embedding_dim, save_path, save_prefix
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

        self.model = ALSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                dropout=self.dropout, n_instruments=self.n_instruments,
                                embedding_dim=self.embedding_dim, n_classes=n_classes).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-5)

    def loss_fn(self, pred, label):
        return nn.functional.cross_entropy(pred, label)

    def compute_metrics(self, preds, labels):
        if len(labels) < 1:
            return np.nan, np.nan
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return accuracy, f1

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        for x_continuous, x_instrument, y_batch in data_loader:
            x_continuous, x_instrument, y_batch = x_continuous.to(self.device), x_instrument.to(
                self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x_continuous, x_instrument)
            loss = self.loss_fn(pred, y_batch)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.optimizer.step()
                losses.append(loss.item())
        return np.mean(losses) if losses else np.nan

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x_continuous, x_instrument, y_batch in data_loader:
                x_continuous, x_instrument, y_batch = x_continuous.to(self.device), x_instrument.to(
                    self.device), y_batch.to(self.device)
                pred = self.model(x_continuous, x_instrument)
                loss = self.loss_fn(pred, y_batch)
                if not torch.isnan(loss): losses.append(loss.item())
        return np.mean(losses) if losses else np.nan

    def fit(self, train_dataset, valid_dataset, batch_size, patience=5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        best_val_loss = float('inf')
        best_param = None
        epochs_no_improve = 0
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            print(f"Epoch {epoch}, train_loss: {train_loss:.6f}, valid_loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss, best_param, epochs_no_improve = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"连续 {patience} 个epoch验证损失没有提升，提前停止。")
                break
        if best_param is not None:
            self.model.load_state_dict(best_param)
            self.fitted = True
            print(f"训练完成，找到最佳模型，验证损失为: {best_val_loss:.6f}")
            model_path = os.path.join(self.save_path, f"{self.save_prefix}.pkl")
            torch.save(best_param, model_path)
            print(f"最优模型已保存至: {model_path}")
        return best_val_loss

    def predict(self, test_dataset, batch_size):
        if not self.fitted: return None, None, np.nan, np.nan
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_preds_classes = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for x_continuous, x_instrument, y_batch in test_loader:
                x_continuous, x_instrument = x_continuous.to(self.device), x_instrument.to(self.device)
                logits = self.model(x_continuous, x_instrument)
                pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds_classes.append(pred_classes)
                all_labels.append(y_batch.numpy())

        all_preds_classes = np.concatenate(all_preds_classes)
        all_labels = np.concatenate(all_labels)

        accuracy, f1 = self.compute_metrics(all_preds_classes, all_labels)

        original_indices = test_dataset.original_df_sorted.iloc[test_dataset.samples].set_index(
            ['datetime', 'instrument']).index
        predictions = pd.Series(all_preds_classes, index=original_indices, name="prediction")

        return predictions, all_labels, accuracy, f1


# %=================================================================
# 任务2: 数据处理主函数
# %=================================================================
def prepare_data_for_tuning(data_path, data_begin_date, valid_start_date, test_start_date, target_label):
    print("--- 开始数据处理 ---")
    data = pd.read_parquet(data_path)

    print(f"本次实验的目标标签为: '{target_label}'")

    all_possible_label_cols = [col for col in data.columns if col.startswith('label_')]
    labels_to_drop = [col for col in all_possible_label_cols if col != target_label]
    print(f"将要删除 {len(labels_to_drop)} 个其他标签列。")
    data = data.drop(columns=labels_to_drop, axis=1)

    data.rename(columns={target_label: 'label'}, inplace=True)

    data['datetime'] = pd.to_datetime(data['datetime'])

    data = data[data['datetime'] >= data_begin_date].copy()
    train_data = data[data['datetime'] < valid_start_date].copy()
    valid_data = data[(data['datetime'] >= valid_start_date) & (data['datetime'] < test_start_date)].copy()
    test_data = data[data['datetime'] >= test_start_date].copy()

    all_dfs = {'train': train_data, 'valid': valid_data, 'test': test_data}
    processed_dfs = {}

    train_data['instrument_idx'], instrument_map = pd.factorize(train_data['instrument'])
    n_instruments = len(instrument_map)
    print(f"从训练集发现 {n_instruments} 个不同的instrument。")
    instrument_to_idx = {name: i for i, name in enumerate(instrument_map)}

    for name, df in all_dfs.items():
        if df.empty:
            processed_dfs[name] = pd.DataFrame()
            continue
        if name != 'train':
            df['instrument_idx'] = df['instrument'].map(instrument_to_idx).fillna(-1).astype(int)

        processed_dfs[name] = df.copy()

    feature_cols = [col for col in train_data.columns if
                    not (col.startswith('label') or col in ['datetime', 'instrument', 'instrument_idx'])]
    d_feat = len(feature_cols)
    print(f"特征维度 (d_feat): {d_feat}")
    print("--- 数据处理完成 ---")
    return processed_dfs['train'], processed_dfs['valid'], processed_dfs['test'], n_instruments, d_feat, instrument_map


# %=================================================================
# 任务3: 单次训练与评估主函数
# %=================================================================
def run_training_session(args, data_cache):
    train_data, valid_data, test_data, n_instruments, d_feat = data_cache['train'], data_cache['valid'], data_cache[
        'test'], data_cache['n_instruments'], data_cache['d_feat']

    # %==================== 修改开始 ====================%
    # 中文注释：这里的核心修改是，我们不再传递原始的目标标签名 (args.target_label)，
    # 而是传递在 prepare_data_for_tuning 函数中统一使用的新名字 'label'。
    # 这样就解决了“列名找不到”的问题。
    train_dataset = TimeSeriesDataset(train_data, target_label_col='label', step_len=args.step_len)
    valid_dataset = TimeSeriesDataset(valid_data, target_label_col='label', step_len=args.step_len)
    # %==================== 修改结束 ====================%

    save_prefix = f"alstm_cls_hs{args.hidden_size}_nl{args.num_layers}_do{args.dropout}_ed{args.embedding_dim}_lr{args.lr}_bs{args.batch_size}"

    model = ALSTMWrapper(
        d_feat=d_feat, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, n_epochs=args.n_epochs, lr=args.lr,
        n_instruments=n_instruments,
        embedding_dim=args.embedding_dim,
        n_classes=3,
        GPU=args.gpu, seed=args.seed, save_path=args.save_path, save_prefix=save_prefix
    )

    val_loss = model.fit(train_dataset, valid_dataset, batch_size=args.batch_size)

    if model.fitted and not test_data.empty:
        # %==================== 修改开始 ====================%
        # 中文注释：同样，为测试集也使用统一的 'label' 列名。
        test_dataset = TimeSeriesDataset(test_data, target_label_col='label', step_len=args.step_len)
        # %==================== 修改结束 ====================%
        _, _, accuracy, f1 = model.predict(test_dataset, batch_size=args.batch_size)
        print(f"\n最终测试集表现 -> Accuracy: {accuracy:.4f}, F1-Score (Macro): {f1:.4f}")
        return val_loss, accuracy, f1
    else:
        print("模型未拟合或测试集为空，跳过测试。")
        return val_loss, np.nan, np.nan


# %=================================================================
# 任务4: 参数解析与主执行流程
# %=================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet'))
    parser.add_argument('--target_label', type=str, default='label_vwap_5m_ud', help='要预测的目标分类标签列名')
    parser.add_argument('--data_begin_date', type=str, default='2024-09-01', help='整个实验使用的数据起始日期')
    parser.add_argument('--valid_start_date', type=str, default='2024-12-01', help='验证集开始日期 (训练集结束日期)')
    parser.add_argument('--test_start_date', type=str, default='2024-12-15', help='固定的最终测试集开始日期')

    parser.add_argument('--step_len', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=os.path.join(base_dir, 'models/ALSTM_tuned_classification/'))

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 在调用函数时，传入本次实验要使用的目标标签。
    train_df, valid_df, test_df, n_ins, d_f, instrument_map = prepare_data_for_tuning(
        data_path=args.data_path,
        data_begin_date=args.data_begin_date,
        valid_start_date=args.valid_start_date,
        test_start_date=args.test_start_date,
        target_label=args.target_label
    )

    data_cache = {"train": train_df, "valid": valid_df, "test": test_df, "n_instruments": n_ins, "d_feat": d_f}

    instrument_map_path = os.path.join(args.save_path, 'instrument_map.pkl')
    with open(instrument_map_path, 'wb') as f:
        pickle.dump(instrument_map, f)
    print(f"Instrument map has been saved to: {instrument_map_path}")

    param_grid = {
        # 'hidden_size': [32, 64],
        #         # 'num_layers': [1, 2],
        #         # 'dropout': [0.2, 0.4],
        #         # 'embedding_dim': [10, 20],
        #         # 'lr': [1e-3, 5e-4, 1e-4],
        #         # 'batch_size': [256, 512, 1024]
        # 'hidden_size': [32],
        # 'num_layers': [1],
        # 'dropout': [0.5],
        # 'embedding_dim': [10],
        # 'lr': [1e-4],
        # 'batch_size': [256]

        # 1. 模型复杂度 (谨慎探索)
        'hidden_size': [32, 64],  # (2个选项) 对比您的稳定基线(32)和一个更复杂的模型(64)

        # 2. 核心正则化参数 (重点探索)
        'dropout': [0.4, 0.5, 0.6],  # (3个选项) 探索比您基线(0.5)更强或稍弱的正则化效果

        # 3. 学习率 (最关键的参数，进行精细搜索)
        'lr': [5e-4, 2e-4, 1e-4],  # (3个选项) 在您已知的稳定值(1e-4)和之前失败的(1e-3)之间进行更细致的搜索

        # 4. Batch Size (探索正则化效果)
        'batch_size': [256, 512],  # (2个选项) 对比您基线(256)和一个正则化效果稍弱、但训练更快的选项

        # 5. 固定参数 (控制变量，简化问题)
        'num_layers': [1],  # (1个选项) 固定为1层，避免模型过于复杂
        'embedding_dim': [10],  # (1个选项) 固定为一个合理值
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    best_val_loss = float('inf')
    best_params = {}
    results_log = []

    for i, params in enumerate(param_combinations):
        print(f"\n{'=' * 30}")
        print(f"正在进行第 {i + 1}/{len(param_combinations)} 组参数搜索: {params}")
        print(f"{'=' * 30}")

        current_args = argparse.Namespace(**vars(args), **params)

        try:
            val_loss, test_accuracy, test_f1 = run_training_session(current_args, data_cache)
            if np.isnan(val_loss):
                print(f"参数组合 {params} 结果为 NaN，跳过。")
                continue

            result_entry = {**params, "val_loss": val_loss, "test_accuracy": test_accuracy, "test_f1_macro": test_f1}
            results_log.append(result_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = result_entry
                print(f"\n*** 发现新的最优验证集损失! Loss: {val_loss:.6f} ***")
                print("当前最优参数:", best_params)

        except Exception as e:
            print(f"参数组合 {params} 运行失败，错误: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n\n--- 超参数搜索完成 ---")
    if best_params:
        print("最佳参数组合 (基于最低验证集损失):")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")

        json_path = os.path.join(args.save_path, "best_params_results.json")
        for key, value in best_params.items():
            if isinstance(value, (np.float32, np.float64)):
                best_params[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                best_params[key] = int(value)

        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"\n最佳参数及测试结果已保存到: {json_path}")

        results_df = pd.DataFrame(results_log)
        results_df.to_csv(os.path.join(args.save_path, "all_tuning_results.csv"), index=False)
        print(f"所有运行日志已保存到: {os.path.join(args.save_path, 'all_tuning_results.csv')}")
    else:
        print("\n所有参数组合均未成功运行，未找到最佳参数。")
