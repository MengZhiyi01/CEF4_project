"""
Futures Price Prediction Demo
============================

A demonstration script for the futures price prediction model using simulated data.
This demo shows the complete pipeline from data generation to model training and evaluation.

Usage:
    python demo.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data import split_data_by_date, create_dataloaders
from src.trainer import Trainer


def generate_synthetic_data(n_samples: int = 10000, n_features: int = 131) -> pd.DataFrame:
    """
    Generate synthetic futures trading data for demonstration.
    
    Args:
        n_samples: Number of data samples
        n_features: Number of features (excluding datetime, instrument, label)
        
    Returns:
        DataFrame with synthetic futures data
    """
    print(f"Generating {n_samples} samples of synthetic futures data...")
    
    # Create time series
    start_date = datetime(2024, 7, 1, 9, 0)
    timestamps = [start_date + timedelta(minutes=i) for i in range(n_samples)]
    
    # Create instrument names
    instruments = ['CU2501', 'AL2501', 'ZN2501', 'AU2501', 'AG2501']
    instrument_list = np.random.choice(instruments, n_samples)
    
    # Generate correlated features
    np.random.seed(42)  # For reproducibility
    
    # Create some underlying factors
    market_trend = np.cumsum(np.random.normal(0, 0.01, n_samples))
    volatility = np.random.exponential(0.02, n_samples)
    
    # Generate features with some structure
    features = []
    for i in range(n_features):
        if i < 20:  # Price-related features
            feature = market_trend + np.random.normal(0, volatility)
        elif i < 50:  # Volume-related features
            feature = np.random.exponential(1000, n_samples) + market_trend * 100
        elif i < 80:  # Technical indicators
            feature = np.random.normal(0, 1, n_samples) + market_trend * 0.5
        else:  # Other features
            feature = np.random.normal(0, 1, n_samples)
        
        features.append(feature)
    
    # Create target variable (future price change)
    # Make it partially predictable from features
    target = (0.3 * features[0] + 0.2 * features[1] + 0.1 * features[10] + 
              np.random.normal(0, 0.5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': timestamps,
        'instrument': instrument_list,
        **{f'feature_{i:03d}': features[i] for i in range(n_features)},
        'label_vwap_5m': target
    })
    
    print(f"Generated data shape: {data.shape}")
    print(f"Time range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Instruments: {data['instrument'].unique()}")
    
    return data


def create_demo_config() -> dict:
    """Create a lightweight configuration for quick demonstration."""
    return {
        'data': {
            'input_path': "demo_data.parquet",
            'train_test_split_date': "2024-07-06",  # Fixed: Within data range
            'train_val_split_ratio': 0.8,
            'datetime_col': "datetime",
            'instrument_col': "instrument",
            'feature_start_idx': 2,
            'feature_end_idx': 133,
            'label_col': "label_vwap_5m",
            'normalize': True,
            'handle_missing': "forward_fill"
        },
        'model': {
            'name': "FNN",
            'hidden_dims': [128, 64, 32],  # Smaller model for quick demo
            'dropout_rate': 0.3,
            'activation': "leaky_relu",  # Use LeakyReLU to avoid dead neurons
            'batch_norm': True
        },
        'training': {
            'batch_size': 256,  # Smaller batch size
            'num_epochs': 20,   # Fewer epochs for quick demo
            'learning_rate': 0.0001,  # Lower learning rate for stability
            'weight_decay': 1e-6,     # Lower weight decay
            'early_stopping_patience': 8,
            'gradient_clip_value': 5.0,  # Higher gradient clipping
            'optimizer': "adam",
            'scheduler': "cosine",
            'scheduler_params': {'T_max': 20}
        },
        'experiment': {
            'name': "futures_demo",
            'seed': 42,
            'save_dir': "./demo_experiments",
            'log_interval': 10,
            'save_best_only': True
        },
        'feature_engineering': {
            'add_time_features': True,
            'add_rolling_features': False,
            'rolling_windows': [5, 10]
        }
    }


def visualize_results(experiment_dir: Path):
    """Create visualization of training results."""
    print("\nGenerating result visualizations...")
    
    # Load predictions
    try:
        val_predictions = pd.read_csv(experiment_dir / 'val_predictions.csv')
        test_predictions = pd.read_csv(experiment_dir / 'test_predictions.csv')
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Futures Price Prediction Demo Results', fontsize=16)
        
        # 1. Validation predictions scatter plot
        axes[0, 0].scatter(val_predictions['true'], val_predictions['pred'], alpha=0.6, s=1)
        axes[0, 0].plot([val_predictions['true'].min(), val_predictions['true'].max()], 
                       [val_predictions['true'].min(), val_predictions['true'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Validation Set Predictions')
        
        # 2. Test predictions scatter plot
        axes[0, 1].scatter(test_predictions['true'], test_predictions['pred'], alpha=0.6, s=1)
        axes[0, 1].plot([test_predictions['true'].min(), test_predictions['true'].max()], 
                       [test_predictions['true'].min(), test_predictions['true'].max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Test Set Predictions')
        
        # 3. Time series plot (subset of test data)
        test_subset = test_predictions.head(500)  # Show first 500 points
        axes[1, 0].plot(test_subset.index, test_subset['true'], label='True', alpha=0.7)
        axes[1, 0].plot(test_subset.index, test_subset['pred'], label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Time Series Comparison (Test Set)')
        axes[1, 0].legend()
        
        # 4. Residuals plot
        residuals = test_predictions['true'] - test_predictions['pred']
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].axvline(0, color='red', linestyle='--')
        
        plt.tight_layout()
        plot_path = experiment_dir / 'demo_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not create visualizations: {e}")


def print_model_summary(experiment_dir: Path):
    """Print summary of model performance."""
    try:
        # Load final metrics
        with open(experiment_dir / 'final_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Best Epoch: {metrics['best_epoch']}")
        print(f"Best Validation Loss: {metrics['best_val_loss']:.6f}")
        print(f"Best Validation IC: {metrics['best_val_ic']:.6f}")
        
        print("\nTest Set Metrics:")
        test_metrics = metrics['test_metrics']
        print(f"  Loss (MSE): {test_metrics['loss']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  IC (Information Coefficient): {test_metrics['ic']:.6f}")
        print(f"  IR (Information Ratio): {test_metrics['ir']:.6f}")
        
        # Interpretation
        print("\nInterpretation:")
        ic_value = test_metrics['ic']
        if abs(ic_value) > 0.05:
            print(f"  IC = {ic_value:.3f}: Good predictive signal")
        elif abs(ic_value) > 0.02:
            print(f"  IC = {ic_value:.3f}: Moderate predictive signal")
        else:
            print(f"  IC = {ic_value:.3f}: Weak predictive signal")
            
        print("="*60)
        
    except Exception as e:
        print(f"Could not load metrics: {e}")


def main():
    """Main demo function."""
    print("FUTURES PRICE PREDICTION MODEL DEMO")
    print("====================================")
    print("This demo demonstrates a complete machine learning pipeline")
    print("for futures price prediction using simulated data.\n")
    
    # 1. Generate synthetic data
    data = generate_synthetic_data(n_samples=10000, n_features=131)
    
    # Save synthetic data
    data_path = Path("demo_data.parquet")
    data.to_parquet(data_path, index=False)
    print(f"Synthetic data saved to: {data_path}")
    
    # 2. Create demo configuration
    config = create_demo_config()
    print(f"\nDemo configuration created:")
    print(f"  Model: {config['model']['hidden_dims']} hidden layers")
    print(f"  Training: {config['training']['num_epochs']} epochs, batch size {config['training']['batch_size']}")
    print(f"  Features: {config['data']['feature_end_idx'] - config['data']['feature_start_idx']} features")
    
    # 3. Split data
    print(f"\nSplitting data by date: {config['data']['train_test_split_date']}")
    train_data, val_data, test_data = split_data_by_date(
        data,
        test_date=config['data']['train_test_split_date'],
        train_val_split_ratio=config['data']['train_val_split_ratio']
    )
    
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Validation set: {len(val_data)} samples")
    print(f"  Test set: {len(test_data)} samples")
    
    # 4. Create data loaders
    print("\nCreating data loaders...")
    feature_cols = [f'feature_{i:03d}' for i in range(131)]
    dataloaders = create_dataloaders(
        train_data,
        val_data,
        test_data,
        feature_cols=feature_cols,
        label_col=config['data']['label_col'],
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Use 0 for demo to avoid multiprocessing issues
        add_time_features=config['feature_engineering']['add_time_features'],
        normalize_by_instrument=True  # 启用按品种标准化
    )
    
    # 5. Train model
    print(f"\nStarting model training...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    trainer = Trainer(config, config['experiment']['name'])
    
    print("\nTraining progress:")
    print("-" * 50)
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    # Save complete model to pickle manually if needed
    try:
        if hasattr(dataloaders, 'get') and 'scaler' in dataloaders and 'feature_names' in dataloaders:
            scaler = dataloaders['scaler']
            feature_names = dataloaders['feature_names']
            
            if scaler is not None and isinstance(feature_names, list):
                # Load best model
                input_dim = next(iter(dataloaders['train']))[0].shape[1]
                best_model = trainer.create_model(input_dim)
                best_model.load_state_dict(torch.load(trainer.experiment_dir / 'best_model.pth'))
                
                # Save to pickle
                pkl_path = trainer.save_model_to_pkl(best_model, scaler, feature_names)
                print(f"\nComplete model package saved to: {pkl_path}")
            else:
                print("Warning: Could not save model to pickle - invalid scaler or feature_names")
        else:
            print("Warning: Could not save model to pickle - scaler or feature_names not available in dataloaders")
    except Exception as e:
        print(f"Warning: Could not save model to pickle: {e}")
    
    # 6. Display results
    print_model_summary(trainer.experiment_dir)
    
    # 7. Create visualizations
    visualize_results(trainer.experiment_dir)
    
    # 8. Cleanup
    print(f"\nDemo completed successfully!")
    print(f"Results saved in: {trainer.experiment_dir}")
    print(f"You can view detailed logs in: {trainer.experiment_dir / 'train.log'}")
    
    # Clean up demo data
    if data_path.exists():
        data_path.unlink()
        print(f"Cleaned up temporary data file: {data_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc() 