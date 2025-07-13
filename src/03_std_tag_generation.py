import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse


def preprocess_and_standardize_data(args):
    """
    一个完整的数据预处理流程，包括加载数据、特征清理、按时间分割、
    时序标准化、增加分类标签以及保存结果。

    Args:
        args (argparse.Namespace): 包含输入和输出路径的命令行参数。
    """
    print("--- 1. 开始加载原始数据 ---")
    if not os.path.exists(args.input_path):
        print(f"错误：输入文件未找到于 {args.input_path}")
        return
    data = pd.read_parquet(args.input_path)
    print(f"原始数据加载完成，形状为: {data.shape}")

    print("\n--- 2. 开始数据清理与特征工程 ---")

    # --- 特征清理：删除已知有问题的列 ---
    columns_to_drop = [
        "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
        "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)"
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"已删除 {len(columns_to_drop)} 个已知问题特征列。")

    # --- 稳健性处理：处理其他可能的inf和nan ---
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    print("数据清理完成。")

    # %==================== 修改(2): 增加涨跌平分类标签 ====================
    #
    # 我们将为每个以'label_'开头的原始收益率列，生成一个新的分类标签列。
    # - 新列名以 '_ud' (up/down) 结尾。
    # - 使用 (2, 1, 0) 编码，这是最适合PyTorch等框架进行分类任务的方式。
    #   - 涨 ( > 0.0005)  -> 2
    #   - 平 ([-0.0005, 0.0005]) -> 1
    #   - 跌 ( < -0.0005)  -> 0
    #
    # =====================================================================
    print("\n--- 3. 开始生成涨跌平分类标签 ---")
    original_label_cols = [col for col in data.columns if col.startswith('label_')]

    for label_col in original_label_cols:
        new_label_col_name = f"{label_col}_ud"
        print(f"正在为 '{label_col}' 创建分类标签 '{new_label_col_name}'...")

        conditions = [
            data[label_col] > args.threshold,
            data[label_col] < -args.threshold
        ]
        choices = [
            2,  # 涨
            0  # 跌
        ]

        data[new_label_col_name] = np.select(conditions, choices, default=1)  # 默认为平

    print(f"已成功生成 {len(original_label_cols)} 个新的分类标签列。")

    print("\n--- 4. 按日期严格切分数据集 (用于标准化) ---")
    data['datetime'] = pd.to_datetime(data['datetime'])
    train_df = data[data['datetime'] < args.valid_start_date].copy()
    valid_df = data[(data['datetime'] >= args.valid_start_date) & (data['datetime'] < args.test_start_date)].copy()
    test_df = data[data['datetime'] >= args.test_start_date].copy()
    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {valid_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    # 动态识别所有特征列
    id_cols = ['datetime', 'instrument']
    all_label_cols = [col for col in data.columns if 'label' in col]  # 包括原始label和新增的_ud label
    feature_cols = [col for col in data.columns if col not in id_cols + all_label_cols]
    print(f"识别出 {len(feature_cols)} 个特征列进行标准化。")

    print("\n--- 5. 开始进行特征标准化 ---")
    scaler = StandardScaler()

    print("正在对训练集进行 fit_transform...")
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])

    print("正在对验证集进行 transform...")
    valid_df.loc[:, feature_cols] = scaler.transform(valid_df[feature_cols])

    print("正在对测试集进行 transform...")
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])
    print("特征标准化完成。")

    # %==================== 修改(1): 输出与原始格式兼容的数据 ====================
    #
    # 我们不再添加 'split' 列，而是将处理好的三个部分重新合并成一个大的DataFrame。
    # 这确保了输出文件的结构与原始输入文件完全一致，可以直接被您现有的
    # alstm_tune4.py 脚本读取和使用，无需任何改动。
    #
    # ========================================================================
    print("\n--- 6. 保存处理后的数据和Scaler对象 ---")
    final_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 保存处理好的数据
    final_df.to_parquet(args.output_path)
    print(f"已将标准化的数据保存至: {args.output_path}")

    # 保存训练好的Scaler对象
    scaler_path = os.path.join(os.path.dirname(args.output_path), 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"已将Scaler对象保存至: {scaler_path}")

    print("\n--- 数据预处理流程全部完成！ ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="增强版数据预处理脚本")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('--input_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_filtered_data_1min_0708v3.parquet'),
                        help='原始数据文件的路径')

    parser.add_argument('--output_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet'),
                        help='处理后数据的保存路径')

    parser.add_argument('--valid_start_date', type=str, default='2024-12-01', help='验证集开始日期')
    parser.add_argument('--test_start_date', type=str, default='2024-12-15', help='测试集开始日期')
    parser.add_argument('--threshold', type=float, default=0.0005, help='定义涨跌的阈值')

    args = parser.parse_args()

    preprocess_and_standardize_data(args)
