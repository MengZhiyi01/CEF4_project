"""
数据加载器工具函数
处理数据分割和DataLoader创建
"""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, Union, Any
from .dataset import FuturesDataset


def split_data_by_date(data: pd.DataFrame, 
                      test_date: str,
                      train_val_split_ratio: float = 0.889) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按日期分割数据集
    
    Args:
        data: 原始数据
        test_date: 测试集开始日期
        train_val_split_ratio: 训练集占训练+验证的比例
        
    Returns:
        train_data, val_data, test_data
    """
    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
        data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 按日期排序
    data = data.sort_values('datetime')
    
    # 分割测试集
    test_mask = data['datetime'] >= pd.to_datetime(test_date)
    test_data = data[test_mask].copy()
    train_val_data = data[~test_mask].copy()
    
    # 分割训练集和验证集
    # 获取唯一的日期
    datetime_series = train_val_data['datetime']
    unique_dates = sorted(datetime_series.dt.date.unique())  # type: ignore
    n_dates = len(unique_dates)
    n_train_dates = int(n_dates * train_val_split_ratio)
    
    train_end_date = unique_dates[n_train_dates - 1]
    train_mask = datetime_series.dt.date <= train_end_date  # type: ignore
    
    train_data = train_val_data[train_mask].copy()
    val_data = train_val_data[~train_mask].copy()
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_data):,} 条数据, "
          f"从 {train_data['datetime'].min()} 到 {train_data['datetime'].max()}")
    print(f"  验证集: {len(val_data):,} 条数据, "
          f"从 {val_data['datetime'].min()} 到 {val_data['datetime'].max()}")
    print(f"  测试集: {len(test_data):,} 条数据, "
          f"从 {test_data['datetime'].min()} 到 {test_data['datetime'].max()}")
    
    return train_data, val_data, test_data  # type: ignore


def create_dataloaders(train_data: pd.DataFrame,
                      val_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      feature_cols: list,
                      label_col: str,
                      batch_size: int = 512,
                      num_workers: int = 4,
                      add_time_features: bool = True,
                      normalize_by_instrument: bool = True,
                      cross_sectional_normalize: bool = False,
                      scale_labels: bool = True) -> Dict[str, Any]:
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        feature_cols: 特征列名列表
        label_col: 标签列名
        batch_size: 批次大小
        num_workers: 数据加载线程数
        add_time_features: 是否添加时间特征
        normalize_by_instrument: 是否按品种分别标准化
        cross_sectional_normalize: 是否使用截面标准化（在每个时间点对所有资产标准化）
        scale_labels: 是否缩放标签
        
    Returns:
        包含train, val, test的DataLoader字典，以及scaler和label_scaler
    """
    # 创建训练集，拟合scaler
    train_dataset = FuturesDataset(
        train_data, 
        feature_cols, 
        label_col,
        scaler=None,
        fit_scaler=True,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        label_scaler=None,
        fit_label_scaler=True,
        scale_labels=scale_labels
    )
    
    # 获取训练集的scaler，如果是截面标准化则为None
    if cross_sectional_normalize or (isinstance(train_dataset.scaler, str) and train_dataset.scaler == "cross_sectional"):
        train_scaler = None
    else:
        # 确保类型安全
        if isinstance(train_dataset.scaler, str):
            train_scaler = None  # 如果意外是字符串，设为None
        else:
            train_scaler = train_dataset.scaler
    
    # 使用训练集的scaler创建验证集和测试集
    val_dataset = FuturesDataset(
        val_data,
        feature_cols,
        label_col,
        scaler=train_scaler,
        fit_scaler=False,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        label_scaler=train_dataset.get_label_scaler(),
        fit_label_scaler=False,
        scale_labels=scale_labels
    )
    
    test_dataset = FuturesDataset(
        test_data,
        feature_cols,
        label_col,
        scaler=train_scaler,
        fit_scaler=False,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        label_scaler=train_dataset.get_label_scaler(),
        fit_label_scaler=False,
        scale_labels=scale_labels
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader创建完成:")
    print(f"  训练集: {len(train_loader)} 个批次")
    print(f"  验证集: {len(val_loader)} 个批次")
    print(f"  测试集: {len(test_loader)} 个批次")
    print(f"  特征维度: {train_dataset.features.shape[1]}")  # type: ignore
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'scaler': train_scaler,  # 使用处理后的scaler
        'label_scaler': train_dataset.get_label_scaler(),
        'feature_names': train_dataset.get_feature_names()
    } 