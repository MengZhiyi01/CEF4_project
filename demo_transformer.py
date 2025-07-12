#!/usr/bin/env python3
"""
Transformer Demo for Futures Price Prediction
A simplified demo version for advisor presentation
Maintains the same architecture as the full training pipeline
"""

import sys
import os
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('./src')
from models import FuturesTransformer
from utils import calculate_ic, calculate_mse, calculate_rmse


class DemoFuturesDataset(Dataset):
    """
    Demo dataset class for futures price prediction
    Same structure as production version but with additional logging for demo
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], label_col: str, 
                 seq_len: int = 20, fit_scaler: bool = False, 
                 feature_scaler=None, label_scaler=None):
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len
        
        print(f"   Dataset Initialization:")
        print(f"   Data shape: {data.shape}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Instruments: {data['instrument'].nunique()}")
        print(f"   Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"   Sequence length: {seq_len}")
        
        # Data preprocessing with same methods as production
        self._preprocess_data(fit_scaler, feature_scaler, label_scaler)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"   Final sequences created: {len(self.sequences)}")
        
    def _preprocess_data(self, fit_scaler: bool, feature_scaler=None, label_scaler=None):
        """Data preprocessing with numerical stability"""
        
        # Feature processing
        features = self.data[self.feature_cols].values.astype(np.float64)
        
        # Handle missing values and outliers
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove extreme outliers (99% percentile clipping)
        for i in range(features.shape[1]):
            col = features[:, i]
            if np.std(col) > 1e-8:
                q99 = np.percentile(col, 99.0)
                q01 = np.percentile(col, 1.0)
                features[:, i] = np.clip(col, q01, q99)
        
        # Feature scaling
        if fit_scaler:
            self.feature_scaler = StandardScaler()
            self.features = self.feature_scaler.fit_transform(features)
        else:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(features) if feature_scaler else features
        
        self.features = self.features.astype(np.float32)
        
        # Label processing
        labels = self.data[self.label_col].values.astype(np.float64)
        
        # Remove label outliers
        label_q99 = np.percentile(labels, 99.0)
        label_q01 = np.percentile(labels, 1.0)
        labels = np.clip(labels, label_q01, label_q99)
        
        if fit_scaler:
            self.label_scaler = StandardScaler()
            self.labels = self.label_scaler.fit_transform(labels.reshape(-1, 1)).flatten()
        else:
            self.label_scaler = label_scaler
            self.labels = self.label_scaler.transform(labels.reshape(-1, 1)).flatten() if label_scaler else labels
        
        self.labels = self.labels.astype(np.float32)
        
        print(f"   Feature stats: mean={np.mean(self.features):.4f}, std={np.std(self.features):.4f}")
        print(f"   Label stats: mean={np.mean(self.labels):.6f}, std={np.std(self.labels):.6f}")
    
    def _create_sequences(self) -> List[Dict]:
        """Create sequence data for time series prediction"""
        sequences = []
        
        # Reset index for proper indexing
        self.data = self.data.reset_index(drop=True)
        
        # Group by instrument and create sequences
        for instrument in self.data['instrument'].unique():
            inst_mask = self.data['instrument'] == instrument
            inst_indices = np.where(inst_mask)[0]
            
            # Sort by datetime
            inst_data_with_idx = [(idx, self.data.loc[idx, 'datetime']) for idx in inst_indices]
            inst_data_with_idx.sort(key=lambda x: x[1])
            sorted_indices = [x[0] for x in inst_data_with_idx]
            
            # Create sequences
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
        
        # Get feature sequence
        feature_indices = seq_info['feature_indices']
        features = self.features[feature_indices]
        
        # Get label
        label = self.labels[seq_info['label_idx']]
        
        # Simple timestamp encoding (position in sequence)
        timestamps = torch.arange(self.seq_len, dtype=torch.long) % 24
        
        # Metadata
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


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_demo_dataloaders(data: pd.DataFrame, feature_cols: List[str], label_col: str,
                           seq_len: int = 20, batch_size: int = 64) -> Dict[str, Any]:
    """Create data loaders for demo (smaller batch size for quick training)"""
    
    # Sort by time
    data = data.sort_values(['datetime', 'instrument']).reset_index(drop=True)
    
    # Time-based split for demo
    unique_dates = sorted(data['datetime'].unique())
    n_dates = len(unique_dates)
    
    # Use 70% for training, 15% for validation, 15% for testing
    train_end_idx = int(n_dates * 0.7)
    val_end_idx = int(n_dates * 0.85)
    
    train_end_date = unique_dates[train_end_idx - 1]
    val_end_date = unique_dates[val_end_idx - 1]
    
    train_data = data[data['datetime'] <= train_end_date].copy()
    val_data = data[(data['datetime'] > train_end_date) & (data['datetime'] <= val_end_date)].copy()
    test_data = data[data['datetime'] > val_end_date].copy()
    
    print(f"   Data Split:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Create datasets
    train_dataset = DemoFuturesDataset(
        train_data, feature_cols, label_col, seq_len, fit_scaler=True
    )
    
    feature_scaler, label_scaler = train_dataset.get_scalers()
    
    val_dataset = DemoFuturesDataset(
        val_data, feature_cols, label_col, seq_len,
        fit_scaler=False, feature_scaler=feature_scaler, label_scaler=label_scaler
    )
    
    test_dataset = DemoFuturesDataset(
        test_data, feature_cols, label_col, seq_len,
        fit_scaler=False, feature_scaler=feature_scaler, label_scaler=label_scaler
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'feature_scaler': feature_scaler,
        'label_scaler': label_scaler
    }


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                device: torch.device, epoch: int) -> Dict[str, float]:
    """Train one epoch with progress tracking"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    
    print(f"  Training Epoch {epoch}...")
    
    for batch_idx, (features, labels, timestamps, metadata) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        timestamps = timestamps.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features, timestamps)
        loss = nn.MSELoss()(outputs.squeeze(), labels.squeeze())
        
        loss.backward()
        
        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        
        # Collect predictions for analysis
        with torch.no_grad():
            predictions.extend(outputs.squeeze().detach().cpu().numpy())
        
        # Show progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.6f}")
    
    predictions = np.array(predictions)
    
    return {
        'loss': total_loss / total_samples,
        'pred_std': np.std(predictions),
        'pred_mean': np.mean(predictions),
        'grad_norm': grad_norm.item() if 'grad_norm' in locals() else 0.0
    }


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                  label_scaler=None, dataset_name: str = 'Test') -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate model performance"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_metadata = []
    total_loss = 0.0
    
    print(f"  Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for features, labels, timestamps, metadata in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            timestamps = timestamps.to(device)
            
            outputs = model(features, timestamps)
            loss = nn.MSELoss()(outputs.squeeze(), labels.squeeze())
            
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.squeeze().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
            
            # Handle metadata
            if isinstance(metadata, dict):
                for i in range(len(metadata['datetime'])):
                    all_metadata.append({
                        'datetime': metadata['datetime'][i],
                        'instrument': metadata['instrument'][i]
                    })
            else:
                all_metadata.extend(metadata)
            
            total_loss += loss.item() * features.size(0)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Transform back to original scale
    if label_scaler is not None:
        original_predictions = label_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        original_labels = label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
    else:
        original_predictions = predictions
        original_labels = labels
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'datetime': [m['datetime'] for m in all_metadata],
        'instrument': [m['instrument'] for m in all_metadata],
        'true_return': original_labels,
        'pred_return': original_predictions
    })
    
    # Calculate metrics
    mse = calculate_mse(labels, predictions)
    rmse = calculate_rmse(labels, predictions)
    
    # Calculate Information Coefficient (IC)
    if np.std(original_predictions) > 1e-10:
        ic, p_value = calculate_ic(original_labels, original_predictions)
    else:
        ic, p_value = 0.0, 1.0
    
    metrics = {
        'loss': total_loss / len(labels),
        'mse': mse,
        'rmse': rmse,
        'ic': ic,
        'ic_p_value': p_value,
        'pred_std': np.std(original_predictions),
        'pred_mean': np.mean(original_predictions),
        'unique_predictions': len(np.unique(np.round(original_predictions, 8)))
    }
    
    # Print results
    print(f"   {dataset_name} Results:")
    print(f"   Loss: {metrics['loss']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   IC: {metrics['ic']:.4f} (p-value: {metrics['ic_p_value']:.4f})")
    print(f"   Prediction Std: {metrics['pred_std']:.6f}")
    print(f"   Unique Predictions: {metrics['unique_predictions']}")
    
    return metrics, results_df


def create_demo_visualization(results_df: pd.DataFrame, save_path: str = 'transformer_examples/demo_results.png'):
    """Create visualization for demo presentation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Transformer Model Demo Results', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs True scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(results_df['true_return'], results_df['pred_return'], alpha=0.6, s=10)
    ax1.plot([results_df['true_return'].min(), results_df['true_return'].max()], 
             [results_df['true_return'].min(), results_df['true_return'].max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Returns')
    ax1.set_ylabel('Predicted Returns')
    ax1.set_title('Prediction vs True Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series of predictions and actuals (sample)
    ax2 = axes[0, 1]
    sample_data = results_df.head(200)  # Show first 200 points
    ax2.plot(sample_data['true_return'], label='True Returns', alpha=0.8)
    ax2.plot(sample_data['pred_return'], label='Predicted Returns', alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Returns')
    ax2.set_title('Time Series Comparison (Sample)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = axes[1, 0]
    residuals = results_df['pred_return'] - results_df['true_return']
    ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by instrument
    ax4 = axes[1, 1]
    instrument_ic = results_df.groupby('instrument').apply(
        lambda x: calculate_ic(x['true_return'], x['pred_return'])[0] if len(x) > 10 else 0
    )
    ax4.bar(range(len(instrument_ic)), instrument_ic.values)
    ax4.set_xlabel('Instrument Index')
    ax4.set_ylabel('Information Coefficient')
    ax4.set_title('IC by Instrument')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   Visualization saved to: {save_path}")


def main():
    """Main demo function"""
    print("=" * 60)
    print("   TRANSFORMER FUTURES PREDICTION DEMO")
    print("=" * 60)
    
    # Set random seed
    set_random_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Load data
    data_files = [
        '../final_filtered_data_1min.parquet',
        './final_filtered_data_1min.parquet'
    ]
    
    data = None
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"  Loading data from: {file_path}")
            data = pd.read_parquet(file_path)
            break
    
    if data is None:
        print("   Data file not found!")
        return
    
    # Use subset for demo (faster training)
    demo_size = 50000  # Use 50k samples for demo
    data = data.head(demo_size)
    print(f"   Using demo dataset: {data.shape[0]} samples")
    
    # Define features and labels
    feature_cols = list(data.columns[2:133])  # Skip datetime, instrument
    label_col = 'label_vwap_5m'
    
    print(f"   Model Configuration:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Label: {label_col}")
    print(f"   Instruments: {data['instrument'].nunique()}")
    
    # Create data loaders
    dataloaders = create_demo_dataloaders(data, feature_cols, label_col, seq_len=20, batch_size=64)
    
    # Create model (same architecture as production)
    model = FuturesTransformer(
        input_dim=len(feature_cols),
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=20,
        dropout=0.1,
        output_dim=1
    ).to(device)
    
    print(f"   Model Architecture:")
    print(f"   {model.get_model_info()}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Quick training for demo (3 epochs)
    print("\n  Starting Demo Training...")
    n_epochs = 3
    
    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{n_epochs} ---")
        
        # Train
        train_stats = train_epoch(model, dataloaders['train'], optimizer, device, epoch)
        
        # Validate
        val_metrics, _ = evaluate_model(model, dataloaders['val'], device, 
                                       dataloaders['label_scaler'], 'Validation')
        
        print(f"  Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_stats['loss']:.6f}")
        print(f"   Val Loss: {val_metrics['loss']:.6f}")
        print(f"   Val IC: {val_metrics['ic']:.4f}")
        print(f"   Prediction Std: {val_metrics['pred_std']:.6f}")
    
    # Final evaluation on test set
    print("\n  Final Test Evaluation:")
    test_metrics, test_results = evaluate_model(model, dataloaders['test'], device,
                                               dataloaders['label_scaler'], 'Test')
    
    # Create visualization
    create_demo_visualization(test_results, 'transformer_examples/transformer_demo_results.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"  Final Results:")
    print(f"   Test IC: {test_metrics['ic']:.4f} (p-value: {test_metrics['ic_p_value']:.4f})")
    print(f"   Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"   Prediction Diversity: {test_metrics['unique_predictions']} unique values")
    print(f"   Model Status: {'  WORKING' if test_metrics['pred_std'] > 1e-6 else '  CONSTANT PREDICTIONS'}")
    
    # Save demo results
    demo_results = {
        'test_metrics': test_metrics,
        'model_info': model.get_model_info(),
        'demo_config': {
            'demo_size': demo_size,
            'n_epochs': n_epochs,
            'features': len(feature_cols),
            'device': str(device)
        }
    }
    
    # Save test results
    test_results.to_csv('transformer_examples/demo_test_results.csv', index=False)
    
    print(f"  Demo results saved to: transformer_examples/demo_test_results.csv")
    print(f"  Visualization saved to: transformer_examples/transformer_demo_results.png")
    
    return demo_results


if __name__ == '__main__':
    main() 