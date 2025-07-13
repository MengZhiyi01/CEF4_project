import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

# 忽略特定于PyTorch的警告，使输出更整洁
warnings.filterwarnings("ignore", category=UserWarning)


# %=================================================================
# 步骤 1: 复用与训练时完全一致的模型和数据集定义
# 这是确保模型能被正确加载和解析的关键。
# %=================================================================

class ALSTMModel(nn.Module):
    """
    ALSTM模型的核心结构。
    (从您的训练脚本 alstm_tune6_ud.py 复制而来，保持完全一致)
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
            input_size=self.hidden_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, dropout=self.dropout,
        )
        self.att_net = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size / 2)),
            nn.Dropout(self.dropout), nn.Tanh(),
            nn.Linear(in_features=int(self.hidden_size / 2), out_features=1, bias=False),
            nn.Softmax(dim=1)
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=self.n_classes)

    def forward(self, x_continuous, x_instrument):
        # 与您的训练脚本完全一致的forward逻辑
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
    自定义的PyTorch数据集类。
    (从您的训练脚本 alstm_tune6_ud.py 复制而来，保持完全一致)
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
        instrument_sample = self.instrument_indices[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        return torch.tensor(features_sample, dtype=torch.float), torch.tensor(instrument_sample,
                                                                              dtype=torch.long), torch.tensor(label,
                                                                                                              dtype=torch.long)


def evaluate_model(args):
    """
    主评估函数，执行所有加载、预测和分析任务。
    """
    print(f"--- 开始评估流程，模型目录: {args.model_dir} ---")

    # --- 步骤 2: 加载所有必要的“组件” ---
    try:
        with open(os.path.join(args.model_dir, 'best_params_results.json'), 'r') as f:
            best_params = json.load(f)
        print("成功加载最优超参数。")
        with open(os.path.join(args.model_dir, 'instrument_map.pkl'), 'rb') as f:
            instrument_map = pickle.load(f)
        print("成功加载品种映射表。")
        model_filename = f"alstm_cls_hs{best_params['hidden_size']}_nl{best_params['num_layers']}_do{best_params['dropout']}_ed{best_params['embedding_dim']}_lr{best_params['lr']}_bs{best_params['batch_size']}.pkl"
        model_path = os.path.join(args.model_dir, model_filename)
        checkpoint_state_dict = torch.load(model_path, map_location=args.device)
        print(f"成功加载模型权重: {model_filename}")
    except FileNotFoundError as e:
        print(f"错误：找不到必要的模型文件: {e}")
        return

    # --- 步骤 3: 准备数据并计算d_feat ---
    print("\n--- 正在准备数据并计算d_feat (与训练脚本逻辑对齐) ---")
    full_data = pd.read_parquet(args.data_path)

    feature_cols = [col for col in full_data.columns if not (
            col.startswith('label') or col in ['datetime', 'instrument', 'instrument_idx']
    )]
    d_feat = len(feature_cols)
    print(f"从数据文件中动态计算出 d_feat: {d_feat}")

    all_label_cols = [col for col in full_data.columns if col.startswith('label_')]
    labels_to_drop = [col for col in all_label_cols if col != args.target_label]
    full_data = full_data.drop(columns=labels_to_drop, axis=1)
    full_data.rename(columns={args.target_label: 'label'}, inplace=True)

    full_data['datetime'] = pd.to_datetime(full_data['datetime'])
    test_data = full_data[full_data['datetime'] >= args.test_start_date].copy()
    if test_data.empty:
        print(f"错误：在起始日期 {args.test_start_date} 之后没有找到测试数据。")
        return

    # %==================== 修改开始 ====================%
    #
    # 中文注释：这里是解决问题的核心。
    # 根据您的最新要求，我们直接从测试集中删除所有在训练时未见过的资产。
    #
    known_instruments = list(instrument_map)  # instrument_map 是一个 pd.Index 对象
    original_test_instruments = set(test_data['instrument'].unique())

    print(f"\n测试集中原始资产数量: {len(original_test_instruments)}")
    test_data = test_data[test_data['instrument'].isin(known_instruments)].copy()
    filtered_test_instruments = set(test_data['instrument'].unique())
    print(f"过滤后 (只保留已知资产) 的资产数量: {len(filtered_test_instruments)}")

    dropped_instruments = original_test_instruments - filtered_test_instruments
    if dropped_instruments:
        print(f"警告：已从测试集中删除 {len(dropped_instruments)} 个未知资产: {dropped_instruments}")
    else:
        print("测试集中未发现新资产，无需删除。")
    # %==================== 修改结束 ====================%

    # --- 步骤 4: 使用正确的参数重建模型并加载权重 ---
    n_instruments_trained = len(instrument_map)

    # %==================== 修改开始 ====================%
    #
    # 中文注释：现在我们可以创建一个与训练时尺寸完全相同的模型，
    # 因为我们已经确保了测试数据中不会有未知资产。
    #
    model = ALSTMModel(
        d_feat=d_feat,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        n_instruments=n_instruments_trained,  # 使用与训练时完全相同的品种数
        embedding_dim=best_params['embedding_dim'],
        n_classes=3
    ).to(args.device)

    # 直接加载权重，不再需要任何复杂的“移植”操作
    model.load_state_dict(checkpoint_state_dict)
    model.eval()
    print("\n模型已成功重建并加载权重。")
    # %==================== 修改结束 ====================%

    # --- 步骤 5: 处理测试集中的品种索引 ---
    instrument_to_idx = {name: i for i, name in enumerate(instrument_map)}
    test_data['instrument_idx'] = test_data['instrument'].map(instrument_to_idx)
    test_data['instrument_idx'] = test_data['instrument_idx'].astype(int)

    # --- 步骤 6: 进行预测 ---
    test_dataset = TimeSeriesDataset(test_data, target_label_col='label', step_len=args.step_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_preds_classes = []
    all_labels = []

    print("\n--- 开始在测试集上进行预测 ---")
    with torch.no_grad():
        for x_continuous, x_instrument, y_batch in test_loader:
            x_continuous, x_instrument = x_continuous.to(args.device), x_instrument.to(args.device)
            logits = model(x_continuous, x_instrument)
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds_classes.append(pred_classes)
            all_labels.append(y_batch.numpy())

    all_preds_classes = np.concatenate(all_preds_classes)
    all_labels = np.concatenate(all_labels)

    results_df = test_dataset.original_df_sorted.iloc[test_dataset.samples].copy()
    results_df['prediction'] = all_preds_classes

    # --- 功能 1: 输出预测值CSV文件 ---
    output_predictions_df = results_df[['datetime', 'instrument', 'prediction', 'label']]
    output_predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    output_predictions_df.to_csv(output_predictions_path, index=False)
    print(f"\n功能 (1): 预测结果已保存至: {output_predictions_path}")

    # --- 功能 2: 计算并输出分资产的评估指标 ---
    print("\n--- 功能 (2): 计算分资产评估指标 ---")
    per_asset_metrics = []
    for instrument, group in results_df.groupby('instrument'):
        accuracy = accuracy_score(group['label'], group['prediction'])
        f1 = f1_score(group['label'], group['prediction'], average='macro', zero_division=0)
        per_asset_metrics.append({'instrument': instrument, 'accuracy': accuracy, 'f1_macro': f1})

    per_asset_metrics_df = pd.DataFrame(per_asset_metrics)

    mean_accuracy = per_asset_metrics_df['accuracy'].mean()
    mean_f1 = per_asset_metrics_df['f1_macro'].mean()

    print("\n各资产评估指标:")
    print(per_asset_metrics_df.to_string())
    print("\n------------------------------------")
    print(f"所有资产平均 Accuracy: {mean_accuracy:.4f}")
    print(f"所有资产平均 F1-Score (Macro): {mean_f1:.4f}")
    print("------------------------------------")

    output_metrics_path = os.path.join(args.output_dir, 'per_asset_metrics.csv')
    per_asset_metrics_df.to_csv(output_metrics_path, index=False)
    print(f"分资产评估指标已保存至: {output_metrics_path}")

    # --- 功能 3: 计算并输出混淆矩阵 ---
    print("\n--- 功能 (3): 计算整体混淆矩阵 ---")
    cm = confusion_matrix(results_df['label'], results_df['prediction'], labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm,
                         index=['真实: 跌(0)', '真实: 平(1)', '真实: 涨(2)'],
                         columns=['预测: 跌(0)', '预测: 平(1)', '预测: 涨(2)'])

    print("混淆矩阵:")
    print(cm_df)

    print("\n--- 评估流程全部完成！ ---")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    parser = argparse.ArgumentParser(description="ALSTM 分类模型评估脚本")

    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(base_dir, '../experiments/models/ALSTM_tuned_classification/'),
                        help='包含最优模型、参数和映射表的目录路径')

    parser.add_argument('--data_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet'),
                        help='预处理后的数据文件路径')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(base_dir, 'evaluation_results/'),
                        help='保存评估结果（CSV文件）的目录')

    parser.add_argument('--target_label', type=str, default='label_vwap_5m_ud', help='要评估的目标分类标签列名')
    parser.add_argument('--test_start_date', type=str, default='2024-12-15', help='与训练时一致的测试集开始日期')
    parser.add_argument('--step_len', type=int, default=10, help='与训练时一致的时间步长')
    parser.add_argument('--batch_size', type=int, default=2048, help='进行预测时的批次大小')

    args = parser.parse_args()

    args.device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_model(args)
