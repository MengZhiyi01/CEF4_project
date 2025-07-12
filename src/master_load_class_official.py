import argparse
import json
import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# 引入 master_tune6_ud.py 中的必要类
# 假设 evaluate_model.py 与 master_tune6_ud.py 在同一目录下
from master_tune6_ud import MASTER, TSDataSampler, DailyBatchSamplerRandom


# --- 定义模型和数据加载器 ---
# 这部分代码直接从 master_tune6_ud.py 复制而来，以确保一致性

class Evaluator:
    """
    用于加载模型并进行评估的类
    """

    def __init__(self, model_path, params, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.model = self.load_model(model_path)
        print(f"模型已成功加载到设备: {self.device}")

    def load_model(self, model_path):
        """加载训练好的PyTorch模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        # 根据参数重新实例化模型结构
        model = MASTER(
            d_feat=self.params['d_feat'],
            d_model=self.params['d_model'],
            t_nhead=self.params['t_nhead'],
            s_nhead=self.params['s_nhead'],
            T_dropout_rate=self.params['dropout'],
            S_dropout_rate=self.params['dropout'],
            num_classes=3
        )

        # 加载状态字典
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # 设置为评估模式
        return model

    def predict(self, data_loader):
        """在给定的数据加载器上进行预测"""
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in data_loader:
                feature = batch_data[:, :, :-1].to(self.device)
                label = batch_data[:, -1, -1].numpy()  # 直接用numpy处理标签

                pred_logits = self.model(feature.float())
                pred = torch.argmax(pred_logits, dim=1).cpu().numpy()

                all_preds.append(pred)
                all_labels.append(label)

        return np.concatenate(all_preds), np.concatenate(all_labels)


def main(args):
    # --- 1. 加载最佳参数和模型 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造最佳参数文件的路径
    params_path = os.path.join(base_dir, args.model_dir, f"{args.prefix}_best_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"找不到最佳参数文件: {params_path}")

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # 动态计算d_feat，以防万一
    # 为了准确性，我们从原始数据加载中获取
    print("加载数据以确定 d_feat...")
    data_path = os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet')
    full_data = pd.read_parquet(data_path)

    # 确定特征列
    id_cols = ['datetime', 'instrument']
    label_cols = [col for col in full_data.columns if 'label' in col]
    feature_cols = [col for col in full_data.columns if col not in id_cols + label_cols]

    # 增加时间特征后的d_feat
    d_feat_calculated = len(feature_cols) + 2
    best_params['d_feat'] = d_feat_calculated
    print(f"从数据中动态确定的 d_feat 为: {d_feat_calculated}")

    # 构造最佳模型文件的路径
    model_filename = f"{args.prefix}_d{best_params['d_model']}_t{best_params['t_nhead']}_s{best_params['s_nhead']}_do{best_params['dropout']}_lr{best_params['lr']}.pt"
    model_path = os.path.join(base_dir, args.model_dir, model_filename)

    print("--- 开始加载模型 ---")
    print(f"参数: {best_params}")
    print(f"模型路径: {model_path}")
    evaluator = Evaluator(model_path, best_params, args.gpu)

    # --- 2. 准备数据 ---
    print("\n--- 准备测试数据 ---")

    # 选择与训练时相同的标签
    labels = [i for i in full_data.columns if i.startswith('label_')]
    labels_to_drop = [i for i in labels if i not in ['label_vwap_5m_ud']]
    data = full_data.drop(columns=labels_to_drop, axis=1)
    data.rename(columns={'label_vwap_5m_ud': 'label'}, inplace=True)

    data['datetime'] = pd.to_datetime(data['datetime'])

    # 增加时间特征
    data['minute_of_day'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute
    minutes_in_day = 24 * 60
    data['time_sin'] = np.sin(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data['time_cos'] = np.cos(2 * np.pi * data['minute_of_day'] / minutes_in_day)
    data = data.drop(columns=['minute_of_day'])

    # 按照训练脚本的日期划分数据
    valid_start_date = '2024-12-01'
    test_start_date = '2024-12-15'

    train_df = data[data['datetime'] < valid_start_date].copy()
    test_df = data[data['datetime'] >= test_start_date].copy()

    # --- 4. 处理新资产问题 ---
    train_assets = set(train_df['instrument'].unique())
    test_assets = set(test_df['instrument'].unique())
    new_assets = test_assets - train_assets

    if new_assets:
        print("\n--- 检测到在测试集中出现但训练集中未见的资产 ---")
        print("这些资产将被正常预测，以检验模型的泛化能力。")
        print(f"未见过的资产列表 ({len(new_assets)}个): {sorted(list(new_assets))}")
    else:
        print("\n--- 测试集中的所有资产均在训练集中出现过 ---")

    # 准备用于TSDataSampler的格式
    for df in [test_df]:
        df['time_id'] = pd.factorize(df['datetime'])[0].astype(np.float32)
        # 注意：此处保留原始的datetime和instrument用于最终输出
        df_sampler = df.drop(columns=['datetime', 'instrument'])
        label_col = df_sampler.pop('label')
        df_sampler.insert(len(df_sampler.columns), 'label', label_col)
        for col in df_sampler.columns:
            if col not in ['time_id']:
                df_sampler[col] = df_sampler[col].astype(np.float32)
        df_sampler.set_index(['time_id', df['instrument']], inplace=True)

    test_min_tid = df_sampler.index.get_level_values('time_id').min()
    test_max_tid = df_sampler.index.get_level_values('time_id').max()

    test_dataset = TSDataSampler(data=df_sampler, start=test_min_tid, end=test_max_tid, step_len=10,
                                 fillna_type='ffill+bfill')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 3. 执行预测 ---
    print("\n--- 在测试集上执行预测 ---")
    predictions, labels = evaluator.predict(test_loader)

    # 将预测结果与原始数据对齐
    # 获取TSDataSampler使用的索引
    sampler_index = test_dataset.get_index()

    results_df = pd.DataFrame({
        'prediction': predictions,
        'label': labels
    }, index=sampler_index)

    # 重置索引以获取 time_id 和 instrument
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'level_1': 'instrument'}, inplace=True)

    # 创建一个从 time_id 到 datetime 的映射
    time_id_to_datetime = test_df[['time_id', 'datetime']].drop_duplicates().set_index('time_id')

    # 将 time_id 映射回 datetime
    results_df['datetime'] = results_df['time_id'].map(time_id_to_datetime['datetime'])

    # 整理最终输出的DataFrame
    final_results_df = results_df[['datetime', 'instrument', 'prediction', 'label']]
    final_results_df.sort_values(by=['datetime', 'instrument'], inplace=True)

    # (1) 保存预测结果到CSV
    output_dir = os.path.join(base_dir, 'evaluation_results_master')
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, 'test_predictions.csv')
    final_results_df.to_csv(predictions_path, index=False)
    print(f"\n(1) 预测结果已保存至: {predictions_path}")

    # --- 4. 计算评估指标 ---
    print("\n--- (2) 计算评估指标 ---")

    # (2a) 分资产计算指标
    asset_metrics = []
    for asset, group in final_results_df.groupby('instrument'):
        pred = group['prediction'].values
        lab = group['label'].values

        if len(lab) < 2: continue

        accuracy = accuracy_score(lab, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(lab, pred, average='macro', zero_division=0)

        asset_metrics.append({
            'instrument': asset,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'num_samples': len(lab)
        })

    asset_metrics_df = pd.DataFrame(asset_metrics)
    metrics_path = os.path.join(output_dir, 'asset_level_metrics.csv')
    asset_metrics_df.to_csv(metrics_path, index=False)
    print(f"分资产评估指标已保存至: {metrics_path}")

    # (2b) 计算并打印平均指标
    if not asset_metrics_df.empty:
        # 使用样本数加权平均
        weighted_avg_metrics = {
            'accuracy': np.average(asset_metrics_df['accuracy'], weights=asset_metrics_df['num_samples']),
            'precision_macro': np.average(asset_metrics_df['precision_macro'], weights=asset_metrics_df['num_samples']),
            'recall_macro': np.average(asset_metrics_df['recall_macro'], weights=asset_metrics_df['num_samples']),
            'f1_macro': np.average(asset_metrics_df['f1_macro'], weights=asset_metrics_df['num_samples']),
        }
        print("\n所有资产的加权平均评估指标:")
        for k, v in weighted_avg_metrics.items():
            print(f"  - {k}: {v:.4f}")

    # --- 5. 计算并打印混淆矩阵 ---
    print("\n--- (3) 整体混淆矩阵 ---")
    cm = confusion_matrix(final_results_df['label'], final_results_df['prediction'], labels=[0, 1, 2])

    print("标签: 0 (跌), 1 (平), 2 (涨)")
    print("行: 真实标签 (True Label), 列: 预测标签 (Predicted Label)")
    print(cm)

    # 可视化混淆矩阵
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"\n混淆矩阵图已保存至: {cm_path}")

    except ImportError:
        print("\n提示: 未安装 seaborn 或 matplotlib，无法生成混淆矩阵图片。")
        print("您可以通过 `pip install seaborn matplotlib` 来安装。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="加载并评估MASTER分类模型")

    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('--model_dir', type=str,
                        default='../experiments/models/MASTER_tuned_classification',
                        help='存放最佳模型和参数文件的目录路径 (相对于脚本位置)')

    parser.add_argument('--prefix', type=str,
                        default='master_clf',
                        help='模型文件名的前缀')

    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')

    parser.add_argument('--batch_size', type=int, default=2048, help='评估时的批处理大小')

    args = parser.parse_args()
    main(args)
