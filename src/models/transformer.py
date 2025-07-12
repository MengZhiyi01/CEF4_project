"""
Transformer model for futures price prediction.
Handles intraday trading patterns and market microstructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0), :]
        return x


class TimeAwareAttention(nn.Module):
    """Time-aware attention mechanism for handling trading sessions."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with time-aware attention."""
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with time-aware attention and feed-forward network."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        self.attention = TimeAwareAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        
        # Feed-forward network
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + attn_output)
        
        return output


class FuturesTransformer(nn.Module):
    """Transformer model for futures price prediction with market microstructure awareness."""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 1024,
                 max_seq_len: int = 100,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        """
        Initialize the Futures Transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimension of model
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Trading session embedding (day/night)
        self.session_embedding = nn.Embedding(3, d_model)  # 0: non-trading, 1: day, 2: night
        
        # Hour embedding for intraday patterns
        self.hour_embedding = nn.Embedding(24, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Create padding mask for variable length sequences."""
        batch_size, seq_len = x.size(0), x.size(1)
        
        if lengths is None:
            return None
            
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(2)
    
    def extract_temporal_features(self, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract trading session and hour from timestamps.
        
        Args:
            timestamps: Tensor of shape (batch_size, seq_len) containing hour values
            
        Returns:
            session_ids: Trading session indicators (0: non-trading, 1: day, 2: night)
            hour_ids: Hour of day (0-23)
        """
        hour_ids = timestamps.long()
        
        # Define trading sessions based on futures market hours
        # Day session: 9:00-11:30, 13:30-15:00
        # Night session: 21:00-02:30
        session_ids = torch.zeros_like(hour_ids)
        
        # Day session
        day_mask = ((hour_ids >= 9) & (hour_ids <= 11)) | ((hour_ids >= 13) & (hour_ids <= 15))
        session_ids[day_mask] = 1
        
        # Night session
        night_mask = (hour_ids >= 21) | (hour_ids <= 2)
        session_ids[night_mask] = 2
        
        return session_ids, hour_ids
    
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            timestamps: Hour timestamps of shape (batch_size, seq_len)
            lengths: Actual sequence lengths for masking
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Add temporal embeddings if timestamps provided
        if timestamps is not None:
            session_ids, hour_ids = self.extract_temporal_features(timestamps)
            session_emb = self.session_embedding(session_ids)
            hour_emb = self.hour_embedding(hour_ids)
            x = x + session_emb + hour_emb
        
        # Create padding mask
        mask = self.create_padding_mask(x, lengths)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Apply output normalization
        x = self.output_norm(x)
        
        # Global average pooling over sequence dimension
        if mask is not None:
            mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            'model_type': 'FuturesTransformer',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 