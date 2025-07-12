#!/usr/bin/env python3
"""
最终稳定版本的Transformer
"""

import sys
import os
import argparse
import random
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add src to path
sys.path.append('./src')
from models import FuturesTransformer
from utils import calculate_ic, calculate_ic_series, calculate_mse, calculate_rmse


class StableFuturesDataset(Dataset):
    """最稳定的期货数据集 - 专注于数值稳定性"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], label_col: str, 
                 seq_len: int = 20, fit_scaler: bool = False, 
                 feature_scaler=None, label_scaler=None):
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len
        
        print(f"StableFuturesDataset initialization:")
        print(f"  Data shape: {data.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Instruments: {data['instrument'].nunique()}")
        print(f"  Sequence length: {seq_len}")
        
        # 极保守的数据预处理
        self._preprocess_data_conservatively(fit_scaler, feature_scaler, label_scaler)
        
        # 创建序列
        self.sequences = self._create_sequences()
        
        print(f"Final sequences created: {len(self.sequences)}")
        
    def _preprocess_data_conservatively(self, fit_scaler: bool, 
                                      feature_scaler=None, label_scaler=None):
        """极保守的数据预处理，确保数值稳定"""
        
        # 特征预处理
        features = self.data[self.feature_cols].values.astype(np.float64)  # 使用double精度
        
        print(f"Original feature stats:")
        print(f"  NaN count: {np.isnan(features).sum()}")
        print(f"  Inf count: {np.isinf(features).sum()}")
        print(f"  Mean: {np.nanmean(features):.6f}")
        print(f"  Std: {np.nanstd(features):.6f}")
        
        # 极保守的异常值处理
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 移除极端异常值（使用更保守的99%分位数）
        for i in range(features.shape[1]):
            col = features[:, i]
            if np.std(col) > 1e-8:  # 避免零方差列
                q99 = np.percentile(col, 99.0)  # 更保守：99% vs 99.9%
                q01 = np.percentile(col, 1.0)   # 更保守：1% vs 0.1%
                features[:, i] = np.clip(col, q01, q99)
        
        # 使用标准标准化（更稳定）
        if fit_scaler:
            self.feature_scaler = StandardScaler()  # 不用RobustScaler，更简单
            self.features = self.feature_scaler.fit_transform(features)
        else:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(features)
        
        # 再次检查缩放后的特征
        self.features = self.features.astype(np.float32)  # 转回float32节省内存
        
        print(f"After feature scaling:")
        print(f"  Mean: {np.mean(self.features):.6f}")
        print(f"  Std: {np.std(self.features):.6f}")
        print(f"  Range: [{np.min(self.features):.6f}, {np.max(self.features):.6f}]")
        print(f"  Max abs value: {np.max(np.abs(self.features)):.6f}")
        
        # 如果缩放后仍有极端值，进一步裁剪
        if np.max(np.abs(self.features)) > 10.0:
            print("  WARNING: Large scaled values detected, applying additional clipping...")
            self.features = np.clip(self.features, -10.0, 10.0)
            print(f"  After clipping: range [{np.min(self.features):.6f}, {np.max(self.features):.6f}]")
        
        # 标签处理 - 也更保守
        labels = self.data[self.label_col].values.astype(np.float64)
        
        # 移除标签异常值
        label_q99 = np.percentile(labels, 99.0)
        label_q01 = np.percentile(labels, 1.0)
        labels = np.clip(labels, label_q01, label_q99)
        
        print(f"Original label stats:")
        print(f"  Mean: {np.mean(labels):.8f}")
        print(f"  Std: {np.std(labels):.8f}")
        print(f"  Range: [{np.min(labels):.8f}, {np.max(labels):.8f}]")
        
        if fit_scaler:
            self.label_scaler = StandardScaler()  # 不用RobustScaler
            self.labels = self.label_scaler.fit_transform(labels.reshape(-1, 1)).flatten()
        else:
            self.label_scaler = label_scaler
            self.labels = self.label_scaler.transform(labels.reshape(-1, 1)).flatten()
        
        self.labels = self.labels.astype(np.float32)
        
        print(f"After label scaling:")
        print(f"  Mean: {np.mean(self.labels):.8f}")
        print(f"  Std: {np.std(self.labels):.8f}")
        print(f"  Range: [{np.min(self.labels):.8f}, {np.max(self.labels):.8f}]")
    
    def _create_sequences(self) -> List[Dict]:
        """创建序列数据"""
        sequences = []
        
        # 重置索引
        self.data = self.data.reset_index(drop=True)
        
        # 按品种分组创建序列
        for instrument in self.data['instrument'].unique():
            inst_mask = self.data['instrument'] == instrument
            inst_indices = np.where(inst_mask)[0]
            
            # 按时间排序
            inst_data_with_idx = [(idx, self.data.loc[idx, 'datetime']) for idx in inst_indices]
            inst_data_with_idx.sort(key=lambda x: x[1])
            sorted_indices = [x[0] for x in inst_data_with_idx]
            
            # 创建序列
            for i in range(len(sorted_indices) - self.seq_len):
                feature_indices = sorted_indices[i:i+self.seq_len]
                label_idx = sorted_indices[i+self.seq_len]
                
                seq_info = {
                    'feature_indices': feature_indices,
                    'label_idx': label_idx,
                    'instrument': instrument,
                    'datetime': str(self.data.loc[label_idx, 'datetime'])
                }
                sequences.append(seq_info)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        
        # 获取特征序列
        feature_indices = seq_info['feature_indices']
        features = self.features[feature_indices]
        
        # 获取标签
        label = self.labels[seq_info['label_idx']]
        
        # 简单的时间戳（使用位置索引）
        timestamps = torch.arange(self.seq_len, dtype=torch.long) % 24
        
        # 元数据
        metadata = {
            'datetime': seq_info['datetime'],
            'instrument': seq_info['instrument']
        }
        
        return (
            torch.FloatTensor(features),
            torch.FloatTensor([label]),
            timestamps,
            metadata
        )
    
    def get_scalers(self):
        return self.feature_scaler, self.label_scaler


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_stable_dataloaders(data: pd.DataFrame, feature_cols: List[str], label_col: str,
                             seq_len: int = 20, batch_size: int = 256,
                             test_split: float = 0.2, val_split: float = 0.1) -> Dict[str, Any]:
    """创建稳定的数据加载器"""
    
    # 按时间排序
    data = data.sort_values(['datetime', 'instrument']).reset_index(drop=True)
    
    # 时间分割
    unique_dates = sorted(data['datetime'].unique())
    n_dates = len(unique_dates)
    
    test_start_idx = int(n_dates * (1 - test_split))
    val_start_idx = int(n_dates * (1 - test_split - val_split))
    
    train_end_date = unique_dates[val_start_idx - 1]
    val_end_date = unique_dates[test_start_idx - 1]
    
    train_data = data[data['datetime'] <= train_end_date].copy()
    val_data = data[(data['datetime'] > train_end_date) & (data['datetime'] <= val_end_date)].copy()
    test_data = data[data['datetime'] > val_end_date].copy()
    
    print(f"Stable data split:")
    print(f"  Train: {len(train_data)} samples ({train_data['datetime'].min()} to {train_data['datetime'].max()})")
    print(f"  Val: {len(val_data)} samples ({val_data['datetime'].min()} to {val_data['datetime'].max()})")
    print(f"  Test: {len(test_data)} samples ({test_data['datetime'].min()} to {test_data['datetime'].max()})")
    
    # 创建数据集
    train_dataset = StableFuturesDataset(
        train_data, feature_cols, label_col, seq_len, 
        fit_scaler=True
    )
    
    feature_scaler, label_scaler = train_dataset.get_scalers()
    
    val_dataset = StableFuturesDataset(
        val_data, feature_cols, label_col, seq_len,
        fit_scaler=False, feature_scaler=feature_scaler, label_scaler=label_scaler
    )
    
    test_dataset = StableFuturesDataset(
        test_data, feature_cols, label_col, seq_len,
        fit_scaler=False, feature_scaler=feature_scaler, label_scaler=label_scaler
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'feature_scaler': feature_scaler,
        'label_scaler': label_scaler
    }


def train_epoch_stable(model: nn.Module, train_loader: DataLoader, 
                      optimizer: torch.optim.Optimizer, device: torch.device, 
                      epoch: int) -> Dict[str, float]:
    """稳定的训练epoch，强化梯度控制"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pred_values = []
    grad_norms = []
    
    for batch_idx, (features, labels, timestamps, metadata) in enumerate(train_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(features, timestamps)
        loss = nn.MSELoss()(outputs.squeeze(), labels.squeeze())
        
        loss.backward()
        
        # 更强的梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更小的阈值
        grad_norms.append(grad_norm.item())
        
        # 检查梯度是否异常
        if grad_norm > 10.0:
            print(f"  WARNING: Large gradient norm {grad_norm:.2f} at batch {batch_idx}")
            optimizer.zero_grad()  # 跳过这个batch
            continue
        
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        
        # 收集预测值统计
        with torch.no_grad():
            pred_values.extend(outputs.squeeze().detach().cpu().numpy())
        
        if batch_idx % 200 == 0:  # 减少打印频率
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.6f}, GradNorm={grad_norm:.6f}")
    
    # 训练统计
    pred_values = np.array(pred_values)
    stats = {
        'loss': total_loss / total_samples,
        'pred_mean': np.mean(pred_values),
        'pred_std': np.std(pred_values),
        'pred_range': np.max(pred_values) - np.min(pred_values),
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_max': np.max(grad_norms)
    }
    
    print(f"  Training stats: pred_std={stats['pred_std']:.6f}, pred_range={stats['pred_range']:.6f}")
    print(f"  Gradient stats: mean_norm={stats['grad_norm_mean']:.6f}, max_norm={stats['grad_norm_max']:.6f}")
    
    return stats


def evaluate_stable(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   label_scaler=None, dataset_name: str = 'val') -> Tuple[Dict[str, float], pd.DataFrame]:
    """稳定的评估函数"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_datetimes = []
    all_instruments = []
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels, timestamps, metadata in data_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)
            
            outputs = model(features, timestamps)
            loss = nn.MSELoss()(outputs.squeeze(), labels.squeeze())
            
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.squeeze().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
            
            # 处理元数据
            if isinstance(metadata, dict):
                batch_size = len(metadata['datetime'])
                for i in range(batch_size):
                    all_datetimes.append(metadata['datetime'][i])
                    all_instruments.append(metadata['instrument'][i])
            else:
                for m in metadata:
                    all_datetimes.append(m['datetime'])
                    all_instruments.append(m['instrument'])
            
            total_loss += loss.item() * features.size(0)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # 预测分析
    print(f"\n{dataset_name} set prediction analysis:")
    print(f"  Predictions: mean={np.mean(predictions):.8f}, std={np.std(predictions):.8f}")
    print(f"  Prediction range: [{np.min(predictions):.8f}, {np.max(predictions):.8f}]")
    print(f"  Unique values: {len(np.unique(np.round(predictions, 10)))}")
    print(f"  Prediction variance: {np.var(predictions):.12f}")
    
    # 检查预测状态
    if np.std(predictions) > 1e-6:
        print(f"   SUCCESS: Predictions are varied!")
        print(f"  Percentiles: p25={np.percentile(predictions, 25):.8f}, p75={np.percentile(predictions, 75):.8f}")
    else:
        print(f"   WARNING: Predictions are still too constant!")
    
    # 反标准化
    original_predictions = predictions.copy()
    original_labels = labels.copy()
    
    if label_scaler is not None:
        original_predictions = label_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        original_labels = label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
        
        print(f"  Original scale: mean={np.mean(original_predictions):.8f}, std={np.std(original_predictions):.8f}")
        print(f"  Original range: [{np.min(original_predictions):.8f}, {np.max(original_predictions):.8f}]")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'datetime': all_datetimes,
        'instrument': all_instruments,
        'true': original_labels,
        'pred': original_predictions
    })
    
    # 计算指标
    mse = calculate_mse(labels, predictions)
    rmse = calculate_rmse(labels, predictions)
    
    # 计算IC（只在预测值有变化时）
    if np.std(original_predictions) > 1e-10:
        overall_ic, overall_p = calculate_ic(original_labels, original_predictions)
        print(f"  IC: {overall_ic:.6f} (p={overall_p:.4f})")
    else:
        overall_ic, overall_p = 0.0, 1.0
        print(f"   Cannot calculate IC: predictions are constant")
    
    metrics = {
        'loss': total_loss / len(labels),
        'mse': mse,
        'rmse': rmse,
        'overall_ic': overall_ic,
        'overall_ic_p_value': overall_p,
        'pred_std': np.std(predictions),
        'pred_range': np.max(predictions) - np.min(predictions),
        'pred_unique_count': len(np.unique(np.round(predictions, 10)))
    }
    
    return metrics, results_df


def main():
    parser = argparse.ArgumentParser(description='Final stable Transformer training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')  # 更小的学习率
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./transformer_final_results', help='Save directory')
    parser.add_argument('--data_subset', type=int, default=0, help='Use data subset (0 for full data)')
    
    args = parser.parse_args()
    
    # 设备设置
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    set_seed(args.seed)
    
    # 加载数据
    data_files = [
        '../final_filtered_data_1min.parquet',
        './final_filtered_data_1min.parquet'
    ]
    
    data = None
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"Loading data from: {file_path}")
            data = pd.read_parquet(file_path)
            break
    
    if data is None:
        raise FileNotFoundError("Data file not found")
    
    # 数据子集
    if args.data_subset > 0:
        data = data.head(args.data_subset)
        print(f"Using data subset: {data.shape}")
    else:
        print(f"Using full dataset: {data.shape}")
    
    # 定义特征和标签
    feature_cols = list(data.columns[2:133])  # 跳过datetime, instrument
    label_col = 'label_vwap_5m'
    
    print(f"Features: {len(feature_cols)}")
    print(f"Label: {label_col}")
    print(f"Unique instruments: {data['instrument'].nunique()}")
    
    # 创建数据加载器
    dataloaders = create_stable_dataloaders(
        data, feature_cols, label_col,
        seq_len=args.seq_len, batch_size=args.batch_size
    )
    
    # 创建更小的模型
    model = FuturesTransformer(
        input_dim=len(feature_cols),
        d_model=64,            # 进一步减少模型大小
        n_heads=4,
        n_layers=2,            # 只用2层
        d_ff=128,              # 减少前馈网络大小
        max_seq_len=args.seq_len,
        dropout=0.3,           # 增加dropout
        output_dim=1
    ).to(device)
    
    print(f"Final model created: {model.get_model_info()}")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.7, verbose=True
    )
    
    # 训练循环
    print("\nStarting final stable training...")
    best_pred_std = 0.0
    best_epoch = 0
    
    training_history = {
        'train_loss': [], 'train_pred_std': [], 'val_loss': [], 'val_pred_std': [],
        'val_ic': [], 'epochs': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        epoch_start = time.time()
        
        # 训练
        train_stats = train_epoch_stable(model, dataloaders['train'], optimizer, device, epoch)
        
        # 验证
        val_metrics, val_results = evaluate_stable(
            model, dataloaders['val'], device, dataloaders['label_scaler'], 'validation'
        )
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        training_history['train_loss'].append(train_stats['loss'])
        training_history['train_pred_std'].append(train_stats['pred_std'])
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_pred_std'].append(val_metrics['pred_std'])
        training_history['val_ic'].append(val_metrics['overall_ic'])
        training_history['epochs'].append(epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats['loss']:.6f}")
        print(f"  Train Pred Std: {train_stats['pred_std']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Pred Std: {val_metrics['pred_std']:.6f}")
        print(f"  Val IC: {val_metrics['overall_ic']:.6f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_metrics['pred_std'] > best_pred_std:
            best_pred_std = val_metrics['pred_std']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model_final.pth')
            print(f"  >>> NEW BEST MODEL! Pred Std: {best_pred_std:.6f}")
        
        # 第一个epoch也保存
        if epoch == 1:
            torch.save(model.state_dict(), 'best_model_final.pth')
    
    # 最终测试
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load('best_model_final.pth', weights_only=True))
    test_metrics, test_results = evaluate_stable(
        model, dataloaders['test'], device, dataloaders['label_scaler'], 'test'
    )
    
    # 保存结果
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    final_results = {
        'model_info': model.get_model_info(),
        'training_args': vars(args),
        'training_history': training_history,
        'best_epoch': best_epoch,
        'best_pred_std': best_pred_std,
        'test_metrics': test_metrics,
        'test_results': test_results,
        'feature_scaler': dataloaders['feature_scaler'],
        'label_scaler': dataloaders['label_scaler']
    }
    
    with open(save_dir / 'final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    test_results.to_csv(save_dir / 'test_predictions_final.csv', index=False)
    
    # 最终总结
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best prediction std: {best_pred_std:.6f}")
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test Pred Std: {test_metrics['pred_std']:.6f}")
    print(f"Test IC: {test_metrics['overall_ic']:.6f}")
    print(f"Test Prediction Range: {test_metrics['pred_range']:.6f}")
    print(f"Test Unique Predictions: {test_metrics['pred_unique_count']}")
    
    if test_metrics['pred_std'] > 1e-6:
        print("\n SUCCESS: Model produces varied predictions!")
        print(" Constant prediction problem SOLVED!")
    else:
        print("\n Issue: Model still produces constant predictions.")
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main() 