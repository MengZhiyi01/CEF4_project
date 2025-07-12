"""
Model trainer for futures price prediction.
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models import FNN
from .utils import calculate_mse, calculate_rmse, calculate_ic, setup_logger


class Trainer:
    """Model trainer for neural network training and evaluation."""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str):
        """
        Args:
            config: Configuration dictionary
            experiment_name: Experiment name
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create experiment directory
        self.experiment_dir = Path(config['experiment']['save_dir']) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.experiment_dir / 'train.log'
        self.logger = setup_logger('trainer', str(log_file))
        
        # TensorBoard
        self.writer = SummaryWriter(self.experiment_dir / 'tensorboard')
        
        # Save configuration
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_ic = -float('inf')
    
    def create_model(self, input_dim: int) -> FNN:
        """Create model."""
        model_config = self.config['model']
        model = FNN(
            input_dim=input_dim,
            hidden_dims=model_config['hidden_dims'],
            output_dim=1,
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation'],
            batch_norm=model_config['batch_norm']
        )
        
        # Log model info
        model_info = model.get_model_info()
        self.logger.info(f"Model info: {model_info}")
        
        # Debug: Check initial model weights
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Check if weights are properly initialized (not all zeros)
        # Note: bias parameters are normally initialized to zero, so we only check weight matrices
        zero_weights = 0
        total_weights = 0
        problematic_layers = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                zero_count = (param.data == 0).sum().item()
                total_count = param.numel()
                zero_weights += zero_count
                total_weights += total_count
                
                # Only warn for weight matrices (not bias), and only if completely zero
                if zero_count == total_count and 'weight' in name:
                    problematic_layers.append(name)
                    self.logger.warning(f"WARNING: Layer {name} has all zero weights!")
        
        if problematic_layers:
            self.logger.warning(f"Found {len(problematic_layers)} layers with all-zero weights: {problematic_layers}")
        else:
            self.logger.info("Model weight initialization looks good")
            
        zero_ratio = zero_weights / total_weights if total_weights > 0 else 0
        self.logger.info(f"Zero parameter ratio: {zero_ratio:.4f} (includes normally-zero bias terms)")
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, Any]:
        """Create optimizer and scheduler."""
        train_config = self.config['training']
        
        # Ensure parameters are correct types
        learning_rate = float(train_config['learning_rate'])
        weight_decay = float(train_config['weight_decay'])
        
        # Optimizer
        if train_config['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'].lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
        
        # Scheduler
        scheduler_type = train_config['scheduler']
        scheduler_params = train_config['scheduler_params']
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_params['T_max']
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_params['step_size'],
                gamma=scheduler_params['gamma']
            )
        elif scheduler_type == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_params['gamma']
            )
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def save_model_to_pkl(self, model: nn.Module, scaler: Any, 
                         feature_names: list, filepath: Optional[str] = None,
                         label_scaler: Optional[Any] = None):
        """
        Save the complete model (including scaler and metadata) to a pickle file.
        
        Args:
            model: Trained model
            scaler: Feature scaler
            feature_names: List of feature names
            filepath: Custom filepath, if None uses default
            label_scaler: Label scaler
        """
        if filepath is None:
            filepath_obj = self.experiment_dir / 'complete_model.pkl'
        else:
            filepath_obj = Path(filepath)
            
        # Create the model package
        model_package = {
            'model_state_dict': model.state_dict(),
            'model_config': self.config['model'],
            'scaler': scaler,
            'label_scaler': label_scaler,
            'feature_names': feature_names,
            'model_class': 'FNN',
            'input_dim': model.input_dim,
            'experiment_config': self.config,
            'best_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_ic': self.best_val_ic
            }
        }
        
        # Save to pickle file
        with open(filepath_obj, 'wb') as f:
            pickle.dump(model_package, f)
            
        self.logger.info(f"Complete model saved to: {filepath_obj}")
        
        # Also save a loading script for convenience
        loading_script = f"""
# Model loading script
import pickle
import torch
import numpy as np
from src.models import FNN

# Load the model package
with open('{filepath_obj}', 'rb') as f:
    model_package = pickle.load(f)

# Recreate the model
model = FNN(
    input_dim=model_package['input_dim'],
    hidden_dims=model_package['model_config']['hidden_dims'],
    output_dim=1,
    dropout_rate=model_package['model_config']['dropout_rate'],
    activation=model_package['model_config']['activation'],
    batch_norm=model_package['model_config']['batch_norm']
)

# Load the state dict
model.load_state_dict(model_package['model_state_dict'])
model.eval()

# Access other components
scaler = model_package['scaler']
label_scaler = model_package.get('label_scaler', None)
feature_names = model_package['feature_names']
experiment_config = model_package['experiment_config']

def predict_with_inverse_transform(model, scaler, label_scaler, feature_names, data):
    \"\"\"Make predictions and inverse transform if label scaler is available.\"\"\"
    # Ensure data has correct features
    if hasattr(data, 'columns'):
        data = data[feature_names]
    
    # Transform features
    if hasattr(scaler, 'transform'):
        data_transformed = scaler.transform(data)
    else:
        # Handle dict-based scaler for per-instrument normalization
        data_transformed = data.copy()
    
    # Make prediction
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data_transformed)
        predictions = model(data_tensor)
    
    predictions_np = predictions.numpy()
    
    # Inverse transform predictions if label_scaler is available
    if label_scaler is not None:
        predictions_np = label_scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()
    
    return predictions_np

print("Model loaded successfully!")
print(f"Model info: {{model.get_model_info()}}")
print(f"Feature names: {{len(feature_names)}} features")
print(f"Label scaler available: {{label_scaler is not None}}")
print(f"Best validation loss: {{model_package['best_metrics']['best_val_loss']:.6f}}")
print(f"Best validation IC: {{model_package['best_metrics']['best_val_ic']:.6f}}")
"""
        
        script_path = filepath_obj.parent / 'load_model.py'
        with open(script_path, 'w') as f:
            f.write(loading_script)
            
        self.logger.info(f"Model loading script saved to: {script_path}")
        
        return filepath_obj
    
    def train_epoch(self, model: nn.Module, 
                   train_loader: DataLoader, 
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        # 在训练开始时检查BatchNorm的running stats
        if epoch == 1:
            self.logger.info("Checking BatchNorm running statistics at epoch 1...")
            for i, bn in enumerate(model.batch_norms):
                running_mean = bn.running_mean.abs().mean().item()
                running_var = bn.running_var.mean().item()
                if running_mean > 1000 or running_var > 1000:
                    self.logger.warning(f"BatchNorm layer {i} has abnormal running stats: mean={running_mean:.2f}, var={running_var:.2f}")
                    # 重置异常的running stats
                    bn.reset_running_stats()
                    self.logger.info(f"Reset BatchNorm layer {i} running statistics")
        
        for batch_idx, (features, labels, metadata) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 检查输入数据是否有异常
            if torch.isnan(features).any() or torch.isinf(features).any():
                self.logger.error(f"NaN or Inf detected in input features at epoch {epoch}, batch {batch_idx}")
                continue
                
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                self.logger.error(f"NaN or Inf detected in labels at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Forward pass
            outputs = model(features)
            
            # 检查模型输出是否有异常
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                self.logger.error(f"NaN or Inf detected in model outputs at epoch {epoch}, batch {batch_idx}")
                continue
            
            loss = nn.MSELoss()(outputs.squeeze(), labels)
            
            # 注意：weight_decay已经在优化器中处理，这里不需要额外添加L2正则化
            # 避免重复计算weight_decay，导致过度正则化
            
            # 检查loss是否异常
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"NaN or Inf detected in loss at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients before clipping
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            # Debug: Check for gradient problems
            if np.isnan(total_grad_norm):
                self.logger.error(f"NaN gradients detected at epoch {epoch}, batch {batch_idx}")
                continue
            elif total_grad_norm > 50:  # 调整阈值，更早发现梯度爆炸
                self.logger.warning(f"Large gradient norm detected: {total_grad_norm:.2f} at epoch {epoch}, batch {batch_idx}")
                
                # 输出更详细的梯度诊断信息
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        if grad_norm > 10:  # 只记录大梯度的层
                            self.logger.warning(f"  Layer {name}: grad_norm={grad_norm:.4f}")
                
                # 当梯度过大时，重置BatchNorm统计
                if hasattr(model, 'reset_running_stats'):
                    model.reset_running_stats()
                    self.logger.info("Reset BatchNorm running statistics due to gradient explosion")
            elif total_grad_norm < 1e-6:
                self.logger.warning(f"Very small gradient norm detected: {total_grad_norm:.2e} at epoch {epoch}, batch {batch_idx}")
                # 检查是否所有梯度都接近0
                zero_grad_layers = []
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.data.norm(2).item() < 1e-8:
                        zero_grad_layers.append(name)
                if zero_grad_layers:
                    self.logger.warning(f"  Layers with near-zero gradients: {zero_grad_layers[:5]}")  # 只显示前5个
                
                # 检查权重健康状况并采取措施
                if not hasattr(self, '_consecutive_small_grads'):
                    self._consecutive_small_grads = 0
                self._consecutive_small_grads += 1
                
                if self._consecutive_small_grads >= 5:  # 连续5个batch梯度消失
                    if hasattr(model, 'check_weight_health'):
                        health = model.check_weight_health()
                        self.logger.warning(f"Weight health check: zero_ratio={health['zero_ratio']:.3f}, small_ratio={health['small_ratio']:.3f}")
                        
                        if health['is_degraded']:
                            # 权重严重退化，重新初始化
                            if hasattr(model, 'reset_weights'):
                                model.reset_weights()
                                self.logger.warning("Model weights reset due to degradation")
                                self._consecutive_small_grads = 0
                        else:
                            # 只重置BatchNorm统计
                            if hasattr(model, 'reset_running_stats'):
                                model.reset_running_stats()
                                self.logger.info("BatchNorm statistics reset due to small gradients")
                                self._consecutive_small_grads = 0
            else:
                # 梯度正常，重置计数器
                if hasattr(self, '_consecutive_small_grads'):
                    self._consecutive_small_grads = 0
            
            # Gradient clipping
            gradient_clip_value = float(self.config['training']['gradient_clip_value'])
            if gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    gradient_clip_value
                )
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            # Log to TensorBoard
            if batch_idx % self.config['experiment']['log_interval'] == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/gradient_norm', total_grad_norm, global_step)
                
                # 额外记录BatchNorm统计信息
                if hasattr(model, 'batch_norms') and len(model.batch_norms) > 0:
                    for i, bn in enumerate(model.batch_norms):
                        running_mean = bn.running_mean.abs().mean().item()
                        running_var = bn.running_var.mean().item()
                        self.writer.add_scalar(f'batchnorm/layer_{i}_running_mean', running_mean, global_step)
                        self.writer.add_scalar(f'batchnorm/layer_{i}_running_var', running_var, global_step)
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self, model: nn.Module, 
                data_loader: DataLoader,
                prefix: str = 'val',
                label_scaler: Optional[Any] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate model."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_metadata = []
        total_loss = 0.0
        
        for features, labels, metadata in data_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = model(features)
            loss = nn.MSELoss()(outputs.squeeze(), labels)
            
            # Collect predictions
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Handle metadata correctly
            if isinstance(metadata, dict):
                # Batch metadata - need to separate into individual samples
                batch_size = len(metadata['datetime'])
                for i in range(batch_size):
                    sample_metadata = {
                        'datetime': metadata['datetime'][i],
                        'instrument': metadata['instrument'][i]
                    }
                    all_metadata.append(sample_metadata)
            else:
                # List of metadata dicts
                all_metadata.extend(metadata)
            
            total_loss += loss.item() * features.size(0)
        
        # Convert to arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Apply label scaling inverse transform if label_scaler is provided
        original_predictions = predictions.copy()
        original_labels = labels.copy()
        
        # Get label scaler from dataset if not provided
        if label_scaler is None:
            try:
                label_scaler = data_loader.dataset.get_label_scaler()  # type: ignore
            except AttributeError:
                pass
        
        if label_scaler is not None:
            # Inverse transform predictions and labels for IC calculation
            original_predictions = label_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            original_labels = label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
            
            print(f"标签反缩放完成:")
            print(f"  缩放标签范围: [{labels.min():.6f}, {labels.max():.6f}]")
            print(f"  原始标签范围: [{original_labels.min():.8f}, {original_labels.max():.8f}]")
            print(f"  缩放预测范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"  原始预测范围: [{original_predictions.min():.8f}, {original_predictions.max():.8f}]")
        
        # Debug: Check if predictions are all the same
        pred_unique = np.unique(predictions)
        if len(pred_unique) == 1:
            self.logger.warning(f"WARNING: All predictions are the same value: {pred_unique[0]}")
            self.logger.warning("This indicates a potential training problem!")
            # 额外的诊断信息
            self.logger.warning(f"Model evaluation mode: {model.training}")
            self.logger.warning(f"Sample input stats - mean: {features[0].mean():.4f}, std: {features[0].std():.4f}")
            # 检查模型各层权重统计
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_max = param.data.abs().max().item()
                    self.logger.warning(f"  {name}: mean={weight_mean:.6f}, std={weight_std:.6f}, max_abs={weight_max:.6f}")
            
            # 检查输入数据的一些样本值
            sample_size = min(5, len(features))
            for i in range(sample_size):
                sample_stats = f"Sample {i}: mean={features[i].mean():.4f}, std={features[i].std():.4f}, range=[{features[i].min():.4f}, {features[i].max():.4f}]"
                self.logger.warning(f"  {sample_stats}")
                
        elif len(pred_unique) < 10:
            self.logger.warning(f"WARNING: Only {len(pred_unique)} unique prediction values found")
            self.logger.warning(f"Unique values: {pred_unique}")
        else:
            self.logger.info(f"Good: {len(pred_unique)} unique prediction values found")
            
        # Debug: Check prediction statistics
        pred_stats = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'unique_count': len(pred_unique)
        }
        self.logger.info(f"Prediction statistics: {pred_stats}")
        
        # Create results DataFrame first - use original scale for interpretability
        results_df = pd.DataFrame({
            'datetime': [m['datetime'] for m in all_metadata],
            'instrument': [m['instrument'] for m in all_metadata],
            'true': original_labels,  # 使用原始标签
            'pred': original_predictions  # 使用原始预测
        })
        
        # For IC calculation, use original scale labels and predictions
        ic_labels = original_labels
        ic_predictions = original_predictions
        
        # Calculate overall metrics using scaled labels for loss computation
        mse = calculate_mse(labels, predictions)
        rmse = calculate_rmse(labels, predictions)
        std_pred = np.std(ic_predictions)  # 使用原始预测的标准差计算IR
        
        # 简化IC计算逻辑，专注于核心指标
        datetime_strs = [m['datetime'] for m in all_metadata]
        instruments = [m['instrument'] for m in all_metadata]
        
        # 1. 计算截面IC（按时间分组计算IC，这是量化投资中的标准做法）
        cross_sectional_ics = []
        unique_datetimes = list(set(datetime_strs))
        
        self.logger.info(f"开始计算截面IC:")
        self.logger.info(f"  总时间点数: {len(unique_datetimes)}")
        self.logger.info(f"  总样本数: {len(ic_labels)}")
        self.logger.info(f"  标签标准差: {np.std(ic_labels):.8f}")
        self.logger.info(f"  预测标准差: {np.std(ic_predictions):.8f}")
        
        # 计算每个时间点的截面IC
        for dt in unique_datetimes:
            dt_indices = [i for i, d in enumerate(datetime_strs) if d == dt]
            
            if len(dt_indices) >= 3:  # 至少需要3个资产
                dt_labels = ic_labels[dt_indices]
                dt_predictions = ic_predictions[dt_indices]
                
                # 检查该时间点的数据质量
                if np.std(dt_labels) > 1e-8 and np.std(dt_predictions) > 1e-8:
                    ic, _ = calculate_ic(dt_labels, dt_predictions)
                    if not np.isnan(ic) and not np.isinf(ic):
                        cross_sectional_ics.append(ic)
        
        # 2. 计算按资产分组的时间序列IC
        instrument_ics = {}
        
        for instrument in set(instruments):
            instrument_indices = [i for i, inst in enumerate(instruments) if inst == instrument]
            
            if len(instrument_indices) >= 10:  # 至少需要10个样本
                inst_labels = ic_labels[instrument_indices]
                inst_predictions = ic_predictions[instrument_indices]
                
                if np.std(inst_labels) > 1e-8 and np.std(inst_predictions) > 1e-8:
                    ic, _ = calculate_ic(inst_labels, inst_predictions)
                    if not np.isnan(ic) and not np.isinf(ic):
                        instrument_ics[instrument] = ic
                    else:
                        instrument_ics[instrument] = 0.0
                else:
                    instrument_ics[instrument] = 0.0
            else:
                instrument_ics[instrument] = 0.0
        
        # 3. 使用截面IC的均值作为主要指标（这是量化投资的标准做法）
        if cross_sectional_ics:
            mean_ic = np.mean(cross_sectional_ics)
            ic_std = np.std(cross_sectional_ics)
            ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
            self.logger.info(f"截面IC统计:")
            self.logger.info(f"  有效截面数: {len(cross_sectional_ics)}/{len(unique_datetimes)}")
            self.logger.info(f"  截面IC均值: {mean_ic:.6f}")
            self.logger.info(f"  截面IC标准差: {ic_std:.6f}")
            self.logger.info(f"  截面IC的IR: {ic_ir:.6f}")
            self.logger.info(f"  正IC比例: {np.sum(np.array(cross_sectional_ics) > 0) / len(cross_sectional_ics):.2%}")
        else:
            mean_ic = 0.0
            self.logger.warning("没有计算出有效的截面IC")
            
            # 如果截面IC失败，尝试使用总体IC作为备选
            if len(ic_labels) > 100:
                overall_ic, _ = calculate_ic(ic_labels, ic_predictions)
                if not np.isnan(overall_ic) and not np.isinf(overall_ic):
                    mean_ic = overall_ic
                    self.logger.info(f"使用总体IC作为备选: {mean_ic:.6f}")
        
        # 保存时间序列IC结果供参考
        time_period_ics = cross_sectional_ics
        
        # Log IC statistics
        if hasattr(self, 'logger'):
            self.logger.info(f"IC calculation results:")
            self.logger.info(f"  截面IC均值: {mean_ic:.6f}")
            self.logger.info(f"  Total samples: {len(labels)}")
            
            if time_period_ics:
                self.logger.info(f"  Time-series IC statistics:")
                self.logger.info(f"    Valid time periods: {len(time_period_ics)} out of {len(unique_datetimes)}")
                self.logger.info(f"    IC mean: {np.mean(time_period_ics):.6f}")
                self.logger.info(f"    IC std: {np.std(time_period_ics):.6f}")
                self.logger.info(f"    IC range: [{np.min(time_period_ics):.6f}, {np.max(time_period_ics):.6f}]")
                positive_rate = np.sum(np.array(time_period_ics) > 0) / len(time_period_ics)
                self.logger.info(f"    Positive IC rate: {positive_rate:.2%}")
            else:
                self.logger.warning(f"No valid time periods for cross-sectional IC calculation")
                self.logger.warning(f"  This may be due to insufficient instruments per time period")
                self.logger.warning(f"  Time periods checked: {len(unique_datetimes)}")
                
                # Show sample time period data
                if len(unique_datetimes) > 0:
                    sample_dt = unique_datetimes[0]
                    sample_indices = [i for i, d in enumerate(datetime_strs) if d == sample_dt]
                    sample_instruments = [instruments[i] for i in sample_indices]
                    self.logger.info(f"  Sample time period {sample_dt}: {len(sample_indices)} instruments: {sample_instruments}")
        
        # 4. 简化的IC稳定性分析
        if time_period_ics and len(time_period_ics) > 1:
            time_series_ic_std = np.std(time_period_ics)
            ic_ir = mean_ic / time_series_ic_std if time_series_ic_std > 0 else 0.0
            self.logger.info(f"  截面IC的IR (信息比率): {ic_ir:.6f}")
            
            # 可选：如果有品种IC数据，显示品种一致性
            if instrument_ics:
                instrument_ic_values = [v for v in instrument_ics.values() if abs(v) > 1e-6]
                if len(instrument_ic_values) > 1:
                    instrument_ic_mean = np.mean(instrument_ic_values)
                    instrument_ic_std = np.std(instrument_ic_values)
                    self.logger.info(f"  品种IC均值: {instrument_ic_mean:.6f}")
                    self.logger.info(f"  品种IC标准差: {instrument_ic_std:.6f}")
        else:
            self.logger.info("  截面IC数据不足，无法计算稳定性指标")
        
        # Calculate IR using mean IC
        ir = float(mean_ic / std_pred) if std_pred > 0 else 0.0
        
        # Log individual instrument performance (time-series IC analysis)
        if hasattr(self, 'logger'):
            meaningful_ics = {k: v for k, v in instrument_ics.items() if abs(v) > 1e-6}
            zero_ics = {k: v for k, v in instrument_ics.items() if abs(v) <= 1e-6}
            
            # Calculate instrument IC statistics
            if meaningful_ics:
                instrument_ic_values = list(meaningful_ics.values())
                self.logger.info(f"Individual instrument IC analysis ({len(meaningful_ics)} out of {len(instrument_ics)} instruments):")
                self.logger.info(f"  Instrument IC mean: {np.mean(instrument_ic_values):.6f}")
                self.logger.info(f"  Instrument IC std: {np.std(instrument_ic_values):.6f}")
                self.logger.info(f"  Best performing instruments (top 5):")
                
                for instrument, ic in sorted(meaningful_ics.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    sample_count = len(results_df[results_df['instrument'] == instrument])
                    self.logger.info(f"    {instrument}: IC={ic:.6f}, samples={sample_count}")
            else:
                self.logger.warning("No instruments have meaningful IC values!")
            
            if zero_ics and len(zero_ics) > 5:
                self.logger.info(f"Instruments with IC≈0 or insufficient data: {len(zero_ics)} instruments")
                # Only log details for instruments with many samples but still zero IC
                problematic = [(k, len(results_df[results_df['instrument'] == k])) 
                              for k in zero_ics.keys() 
                              if len(results_df[results_df['instrument'] == k]) > 100]
                if problematic:
                    self.logger.warning(f"Large instruments with zero IC (potential data issues): {len(problematic)} instruments")
                    # Show only top 3 most problematic
                    for inst, count in sorted(problematic, key=lambda x: x[1], reverse=True)[:3]:
                        self.logger.warning(f"    {inst}: {count} samples")
        
        metrics = {
            'loss': total_loss / len(labels),
            'mse': mse,
            'rmse': rmse,
            'ic': mean_ic,  # 使用时间截面IC的均值（更稳定的IC计算方法）
            'ir': ir,
            'instrument_ics': instrument_ics,  # 保存各品种的时间序列IC
            'time_period_ics': time_period_ics,  # 保存各时间段的截面IC
            'n_valid_periods': len(time_period_ics) if time_period_ics else 0
        }
        
        return metrics, results_df
    
    def train(self, train_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader] = None,
             label_scaler: Optional[Any] = None):
        """Main training loop."""
        
        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]
        
        # Get label scaler from dataset
        if label_scaler is None:
            try:
                label_scaler = train_loader.dataset.get_label_scaler()  # type: ignore
            except AttributeError:
                pass
        
        # Create model
        model = self.create_model(input_dim)
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model)
        
        # Training loop
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            
            # Training
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validation
            val_metrics, val_results = self.evaluate(model, val_loader, 'val', label_scaler)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}: "
                           f"Train Loss: {train_metrics['loss']:.6f}, "
                           f"Val Loss: {val_metrics['loss']:.6f}, "
                           f"Val IC: {val_metrics['ic']:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/ic', val_metrics['ic'], epoch)
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_ic = val_metrics['ic']
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), self.experiment_dir / 'best_model.pth')
                
                # Save validation predictions
                val_results.to_csv(self.experiment_dir / 'val_predictions.csv', index=False)
                
                # Save metrics
                with open(self.experiment_dir / 'best_metrics.json', 'w') as f:
                    json.dump(val_metrics, f, indent=4)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation IC: {self.best_val_ic:.6f}")
        
        # Test evaluation
        if test_loader is not None:
            # Load best model
            model.load_state_dict(torch.load(self.experiment_dir / 'best_model.pth'))
            
            # Test evaluation
            test_metrics, test_results = self.evaluate(model, test_loader, 'test', label_scaler)
            
            # Save test results
            test_results.to_csv(self.experiment_dir / 'test_predictions.csv', index=False)
            
            # Save final metrics
            final_metrics = {
                'best_epoch': best_epoch,
                'best_val_loss': self.best_val_loss,
                'best_val_ic': self.best_val_ic,
                'test_metrics': test_metrics
            }
            
            with open(self.experiment_dir / 'final_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)
            
            self.logger.info("Test metrics:")
            for key, value in test_metrics.items():
                if key in ['instrument_ics', 'time_period_ics']:
                    self.logger.info(f"  {key}: {value}")
                else:
                    self.logger.info(f"  {key}: {value:.6f}")
        
        self.writer.close()
        
        # Save complete model to pickle file
        if test_loader is not None:
            try:
                # Get scaler and feature names from dataloader
                scaler = getattr(test_loader.dataset, 'scaler', None)
                feature_names = getattr(test_loader.dataset, 'feature_names', getattr(test_loader.dataset, 'feature_cols', []))
                
                if scaler is not None and feature_names:
                    # Load best model for saving
                    best_model = self.create_model(input_dim)
                    best_model.load_state_dict(torch.load(self.experiment_dir / 'best_model.pth'))
                    
                    # Save to pickle
                    pkl_path = self.save_model_to_pkl(best_model, scaler, feature_names, label_scaler=label_scaler)
                    self.logger.info(f"Complete model package saved to: {pkl_path}")
                else:
                    self.logger.warning("Could not save model to pickle: scaler or feature_names not found")
            except Exception as e:
                self.logger.error(f"Error saving model to pickle: {e}")
        
        self.logger.info("Training completed.") 