"""
Feedforward Neural Network (FNN) model for futures price prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import numpy as np


class FNN(nn.Module):
    """Feedforward Neural Network model for regression tasks."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int = 1,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default 1 for regression)
            dropout_rate: Dropout rate
            activation: Activation function type
            batch_norm: Whether to use batch normalization
        """
        super(FNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation_type = activation
        self.use_batch_norm = batch_norm
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                # 使用更合理的BatchNorm参数，避免统计更新过慢
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1))  # 恢复到默认momentum值
                
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.01)  # 添加LeakyReLU选项，避免死神经元
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _init_weights(self):
        """Initialize model weights with improved strategy."""
        # 使用更激进的初始化策略来避免梯度消失
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # 对LeakyReLU使用He初始化
                if self.activation_type.lower() == 'leaky_relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif self.activation_type.lower() == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # 对其他激活函数使用Xavier初始化，但增加方差
                    fan_in = layer.weight.shape[1]
                    fan_out = layer.weight.shape[0]
                    # 增加初始化范围，确保足够的信号传播
                    bound = np.sqrt(6.0 / (fan_in + fan_out)) * 1.5  # 增加1.5倍
                    nn.init.uniform_(layer.weight, -bound, bound)
                
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # 初始化BatchNorm参数 - 确保初始时有足够的方差
        for bn in self.batch_norms:
            nn.init.constant_(bn.weight, 1.0)  # 保持原始方差
            nn.init.constant_(bn.bias, 0.0)
            
        # 输出层使用较大的初始化，确保初始预测有一定的方差
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.2)  # 进一步增加到0.2
        if self.output_layer.bias is not None:
            # 给输出层bias一个小的初始值，避免完全从0开始
            nn.init.normal_(self.output_layer.bias, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = self.dropouts[i](x)
        
        # Output layer (no activation for regression)
        x = self.output_layer(x)
        
        return x
    
    def reset_running_stats(self):
        """Reset BatchNorm running statistics."""
        for bn in self.batch_norms:
            bn.reset_running_stats()
            
    def reset_weights(self):
        """重新初始化所有权重"""
        print("警告：检测到权重退化，重新初始化模型权重")
        self._init_weights()
        # 也重置BatchNorm统计
        self.reset_running_stats()
        
    def check_weight_health(self):
        """检查权重健康状况"""
        total_params = 0
        zero_params = 0
        very_small_params = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                param_data = param.data.abs()
                total_params += param_data.numel()
                zero_params += (param_data == 0).sum().item()
                very_small_params += (param_data < 1e-6).sum().item()
        
        zero_ratio = zero_params / total_params if total_params > 0 else 0
        small_ratio = very_small_params / total_params if total_params > 0 else 0
        
        # 如果超过50%的权重是0或极小值，认为权重退化
        return {
            'zero_ratio': zero_ratio,
            'small_ratio': small_ratio,
            'is_degraded': zero_ratio > 0.5 or small_ratio > 0.8
        }
    
    def get_regularization_loss(self, l1_lambda: float = 0.0, l2_lambda: float = 0.0) -> torch.Tensor:
        """
        Calculate regularization loss.
        
        Args:
            l1_lambda: L1 regularization coefficient
            l2_lambda: L2 regularization coefficient
            
        Returns:
            Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for param in self.parameters():
            if l1_lambda > 0:
                reg_loss += l1_lambda * torch.norm(param, 1)
            if l2_lambda > 0:
                reg_loss += l2_lambda * torch.norm(param, 2) ** 2
                
        return reg_loss
    
    def get_feature_importance(self, dataloader: Any, device: str = 'cpu') -> np.ndarray:
        """
        Calculate feature importance using gradient method.
        
        Args:
            dataloader: Data loader
            device: Computation device
            
        Returns:
            Feature importance array
        """
        self.eval()
        feature_importance = torch.zeros(self.input_dim).to(device)
        total_samples = 0
        
        with torch.enable_grad():
            for batch in dataloader:
                features, labels, _ = batch
                features = features.to(device).requires_grad_(True)
                labels = labels.to(device)
                
                # Forward pass
                outputs = self(features)
                loss = F.mse_loss(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradient magnitude as importance
                feature_importance += torch.abs(features.grad).sum(dim=0)
                total_samples += features.size(0)
                
                # Clear gradients
                features.grad.zero_()
        
        # Average and convert to numpy
        feature_importance = (feature_importance / total_samples).cpu().numpy()
        
        return feature_importance
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'batch_norm': self.use_batch_norm,
            'total_parameters': self.count_parameters()
        } 