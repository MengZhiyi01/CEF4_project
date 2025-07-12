"""
Training script for futures price prediction model.
"""

import sys
import os
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import split_data_by_date, create_dataloaders
from src.trainer import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train futures price prediction model')
    parser.add_argument('--config', type=str, default='config/config_cross_sectional.json',
                       help='Configuration file path')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Data file path (overrides config)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}")
    
    # Load configuration
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif args.config.endswith('.json'):
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {args.config}")
    
    # Override configuration
    if args.data_path:
        config['data']['input_path'] = args.data_path
    if args.seed:
        config['experiment']['seed'] = args.seed
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = config['experiment']['name'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"Experiment: {experiment_name}")
    
    # Load data
    print("Loading data...")
    data_path = Path(config['data']['input_path'])
    if not data_path.exists():
        # Try relative path
        data_path = Path(__file__).parent / config['data']['input_path']
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    data = pd.read_parquet(data_path)
    print(f"Data shape: {data.shape}")
    
    # Get feature columns - extract technical indicator columns (2-133)
    feature_cols = list(data.columns[config['data']['feature_start_idx']:config['data']['feature_end_idx']])
    label_col = config['data']['label_col']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns (first 5): {feature_cols[:5]}")
    print(f"Feature columns (last 5): {feature_cols[-5:]}")
    print(f"Label column: {label_col}")
    
    # Verify feature columns are correct (should be technical indicators)
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check feature_start_idx and feature_end_idx in config.")
    
    # Print data statistics for verification
    print(f"Data statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Unique instruments: {data['instrument'].nunique()}")
    print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"  Label statistics: mean={data[label_col].mean():.6f}, std={data[label_col].std():.6f}")
    
    # Split data
    print("Splitting data...")
    train_data, val_data, test_data = split_data_by_date(
        data,
        test_date=config['data']['train_test_split_date'],
        train_val_split_ratio=config['data']['train_val_split_ratio']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    
    # 检查是否使用截面标准化
    use_cross_sectional = config.get('feature_engineering', {}).get('cross_sectional_normalize', False)
    
    dataloaders = create_dataloaders(
        train_data,
        val_data,
        test_data,
        feature_cols=feature_cols,
        label_col=label_col,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        add_time_features=config['feature_engineering']['add_time_features'],
        normalize_by_instrument=not use_cross_sectional,  # 如果使用截面标准化则不按品种标准化
        cross_sectional_normalize=use_cross_sectional,  # 启用截面标准化
        scale_labels=True  # 启用标签缩放
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(config, experiment_name)
    
    # Start training
    print("Starting training...")
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        label_scaler=dataloaders['label_scaler']
    )
    
    print("Training completed.")
    print(f"Results saved to: {trainer.experiment_dir}")


if __name__ == '__main__':
    main() 