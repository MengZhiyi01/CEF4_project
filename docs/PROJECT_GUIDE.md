# Futures Price Prediction Project Guide

## Overview

This guide provides a comprehensive walkthrough of the futures price prediction project for academic advisors and researchers. The project implements advanced deep learning techniques for short-term futures price forecasting using high-frequency trading data from Chinese commodity markets.

## Project Structure

```
futures_prediction/
├── src/                           # Core source code
│   ├── models/                    # Model implementations
│   │   ├── transformer.py         # Transformer architecture
│   │   ├── fnn.py                 # Feed-forward neural network
│   │   └── __init__.py
│   ├── utils/                     # Utility functions
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── logger.py              # Logging utilities
│   │   └── __init__.py
│   ├── data/                      # Data processing modules
│   └── trainer.py                 # Training orchestrator
├── experiments/                   # Experiment results
│   ├── transformer_final_stable/  # Best model results
│   └── [other experiments]/
├── config/                        # Configuration files
│   ├── config_transformer.json   # Transformer config
│   ├── config_demo.json          # Demo configuration
│   └── [other configs]/
├── docs/                          # Documentation
│   ├── ACADEMIC_PROJECT_OVERVIEW.md
│   ├── PROJECT_GUIDE.md          # This file
│   └── [other docs]/
├── train_transformer.py          # Main training script
├── demo_transformer.py           # Demo script
├── train_fnn.py                  # FNN training script
├── requirements.txt              # Python dependencies
└── README.md                     # Project README
```

## Project Logic and Methodology

### 1. Data Processing Pipeline

#### 1.1 Data Input
- **Source**: Chinese futures market 1-minute data
- **Format**: Parquet files with 137 columns
- **Size**: ~2M records covering 73 contracts
- **Period**: July 2024 to December 2024

#### 1.2 Feature Engineering
The project creates 131 features from raw market data:

```python
# Feature categories
price_features = ['open', 'high', 'low', 'close', 'vwap']
volume_features = ['volume', 'turnover', 'open_interest']
technical_indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd']
microstructure_features = ['bid_ask_spread', 'order_imbalance']
volatility_features = ['realized_vol', 'garch_vol']
```

#### 1.3 Data Preprocessing
1. **Outlier Handling**: 99th percentile clipping
2. **Missing Value Treatment**: Forward fill and interpolation
3. **Normalization**: StandardScaler on features
4. **Sequence Construction**: 20-step sliding windows

#### 1.4 Label Construction
```python
# 5-minute forward VWAP return
label = (future_vwap_5min - current_close) / current_close
```

### 2. Model Architecture

#### 2.1 Transformer Model
The core innovation is a time-aware Transformer architecture:

```python
# Key components
- Input projection: 131 → 64 dimensions
- Positional encoding: Sinusoidal + temporal
- Trading session embedding: Day/night/non-trading
- Multi-head attention: 4 heads
- Feed-forward network: 128 hidden units
- Output projection: Global pooling → 1 output
```

#### 2.2 Baseline FNN Model
```python
# Architecture
- Input: 131 features
- Hidden layers: [256, 128, 64]
- Activation: ReLU
- Regularization: Dropout + BatchNorm
- Output: 1 regression target
```

### 3. Training Strategy

#### 3.1 Numerical Stability Approach
The project solved the critical "constant prediction" problem:

```python
# Stability measures
1. Conservative gradient clipping (max_norm=0.5)
2. Double precision preprocessing
3. Prediction variance monitoring
4. Adaptive learning rate scheduling
```

#### 3.2 Training Configuration
```python
# Hyperparameters
optimizer = AdamW(lr=0.00005, weight_decay=0.01)
scheduler = ReduceLROnPlateau(patience=2, factor=0.7)
batch_size = 256
max_epochs = 10
gradient_clip = 0.5
```

### 4. Evaluation Framework

#### 4.1 Primary Metrics
- **Information Coefficient (IC)**: Correlation between predictions and returns
- **IC p-value**: Statistical significance testing
- **RMSE**: Root mean squared error
- **Prediction diversity**: Unique prediction count

#### 4.2 Success Criteria
```python
# Model success indicators
ic_threshold = 0.02          # Minimum IC for significance
p_value_threshold = 0.05     # Statistical significance
pred_std_threshold = 1e-6    # Avoid constant predictions
unique_pred_threshold = 1000 # Prediction diversity
```

## Running the Project

### 1. Environment Setup

#### 1.1 System Requirements
- **Python**: 3.8+
- **CUDA**: Optional but recommended for GPU acceleration
- **Memory**: 8GB+ RAM
- **Storage**: 2GB+ for data and models

#### 1.2 Installation
```bash
# Clone repository (if from git)
git clone [repository_url]
cd futures_prediction

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 2. Data Preparation

#### 2.1 Data Location
Ensure the data file is available:
```bash
# Expected data file locations
./final_filtered_data_1min.parquet
../final_filtered_data_1min.parquet
```

#### 2.2 Data Validation
```python
# Check data structure
python -c "
import pandas as pd
data = pd.read_parquet('final_filtered_data_1min.parquet')
print(f'Data shape: {data.shape}')
print(f'Columns: {list(data.columns[:10])}...')
print(f'Date range: {data[\"datetime\"].min()} to {data[\"datetime\"].max()}')
"
```

### 3. Model Training

#### 3.1 Quick Demo (5 minutes)
```bash
# Run demo for advisor presentation
python demo_transformer.py

# Expected output:
# - Training progress (3 epochs)
# - Validation metrics
# - Test results with IC calculation
# - Visualization generation
```

#### 3.2 Full Training (30-60 minutes)
```bash
# Full transformer training
python train_transformer.py --epochs 10 --batch_size 256

# With custom configuration
python train_transformer.py --config config/config_transformer.json

# Monitor training progress
tail -f experiments/[experiment_name]/train.log
```

#### 3.3 Baseline Model
```bash
# Train FNN baseline
python train_fnn.py --epochs 50 --batch_size 512
```

### 4. Configuration Options

#### 4.1 Transformer Configuration
```json
{
    "model": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.3
    },
    "training": {
        "batch_size": 256,
        "learning_rate": 0.00005,
        "num_epochs": 10,
        "gradient_clip_value": 0.5
    }
}
```

#### 4.2 Command Line Arguments
```bash
# Key arguments
--epochs N          # Number of training epochs
--batch_size N      # Batch size for training
--lr FLOAT         # Learning rate
--seq_len N        # Sequence length (default: 20)
--gpu N            # GPU device number
--seed N           # Random seed
--save_dir PATH    # Results save directory
```

### 5. Model Evaluation

#### 5.1 Automated Evaluation
The training script automatically evaluates the model:

```python
# Metrics calculated
- Training loss progression
- Validation IC and p-value
- Test set performance
- Prediction diversity analysis
- Statistical significance testing
```

#### 5.2 Results Interpretation

**Good Results:**
```
Test IC: 0.0314 (p-value: 0.0001)
Test Prediction Std: 0.074
Unique Predictions: 402,145
Status: ✅ WORKING
```

**Problematic Results:**
```
Test IC: 0.001 (p-value: 0.89)
Test Prediction Std: 0.0000001
Unique Predictions: 5
Status: ❌ CONSTANT PREDICTIONS
```

### 6. Output Files

#### 6.1 Training Results
```
experiments/[experiment_name]/
├── best_model.pth              # Best model weights
├── training_history.json       # Training metrics
├── test_predictions.csv        # Detailed predictions
├── config.json                 # Used configuration
└── train.log                   # Training log
```

#### 6.2 Demo Results
```
demo_test_results.csv           # Test predictions
transformer_demo_results.png    # Visualization
```

## Understanding the Results

### 1. Key Performance Indicators

#### 1.1 Information Coefficient (IC)
- **Definition**: Correlation between predictions and actual returns
- **Good Values**: IC > 0.02 with p-value < 0.05
- **Interpretation**: Measures predictive power

#### 1.2 Prediction Diversity
- **Metric**: Standard deviation of predictions
- **Good Values**: Std > 0.01
- **Interpretation**: Ensures model isn't stuck in constant predictions

#### 1.3 Statistical Significance
- **Test**: T-test of IC values against zero
- **Threshold**: p-value < 0.05
- **Interpretation**: Confirms results aren't due to random chance

### 2. Model Comparison

#### 2.1 Expected Performance Hierarchy
```
Transformer (time-aware) > FNN (baseline) > Random
Target IC: 0.03           Target IC: 0.02    IC: ~0.00
```

#### 2.2 Training Time Comparison
```
Demo (3 epochs):     ~3 minutes
Full Training:       ~30 minutes
FNN Training:        ~10 minutes
```

### 3. Troubleshooting

#### 3.1 Common Issues

**Issue**: Constant predictions
```
Solution: 
- Check gradient clipping is enabled
- Monitor prediction variance
- Verify data preprocessing
```

**Issue**: Poor IC performance
```
Solution:
- Increase model complexity
- Adjust learning rate
- Check data quality
```

**Issue**: Training instability
```
Solution:
- Reduce learning rate
- Increase gradient clipping
- Check for data leakage
```

#### 3.2 Debugging Commands
```bash
# Check model architecture
python -c "
from src.models import FuturesTransformer
model = FuturesTransformer(131)
print(model.get_model_info())
"

# Validate data loading
python -c "
from train_transformer import create_stable_dataloaders
import pandas as pd
data = pd.read_parquet('final_filtered_data_1min.parquet')
loaders = create_stable_dataloaders(data, list(data.columns[2:133]), 'label_vwap_5m')
print(f'Train batches: {len(loaders[\"train\"])}')
"
```

## Technical Implementation Details

### 1. Data Processing Pipeline

#### 1.1 Sequence Creation
```python
# Time-aware sequence construction
def create_sequences(data, seq_len=20):
    sequences = []
    for instrument in data['instrument'].unique():
        inst_data = data[data['instrument'] == instrument]
        for i in range(len(inst_data) - seq_len):
            x = inst_data.iloc[i:i+seq_len]
            y = inst_data.iloc[i+seq_len]
            sequences.append((x, y))
    return sequences
```

#### 1.2 Feature Scaling
```python
# Robust scaling approach
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.astype(np.float64))
features_scaled = np.clip(features_scaled, -10.0, 10.0)
```

### 2. Model Implementation

#### 2.1 Time-Aware Attention
```python
class TimeAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.session_embedding = nn.Embedding(3, d_model)
    
    def forward(self, x, timestamps):
        # Extract trading session info
        session_ids = self.extract_sessions(timestamps)
        session_emb = self.session_embedding(session_ids)
        
        # Enhanced attention with session awareness
        x_enhanced = x + session_emb
        return self.attention(x_enhanced, x_enhanced, x_enhanced)
```

#### 2.2 Numerical Stability
```python
# Training stability measures
def stable_training_step(model, batch, optimizer):
    optimizer.zero_grad()
    
    loss = compute_loss(model, batch)
    loss.backward()
    
    # Conservative gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Skip if gradients explode
    if grad_norm > 10.0:
        optimizer.zero_grad()
        return None
    
    optimizer.step()
    return loss.item()
```

### 3. Evaluation Implementation

#### 3.1 IC Calculation
```python
def calculate_ic(y_true, y_pred):
    from scipy.stats import pearsonr
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate correlation
    if len(y_true_clean) < 2:
        return 0.0, 1.0
    
    ic, p_value = pearsonr(y_true_clean, y_pred_clean)
    return ic, p_value
```

#### 3.2 Prediction Quality Assessment
```python
def assess_prediction_quality(predictions):
    return {
        'variance': np.var(predictions),
        'std': np.std(predictions),
        'range': np.max(predictions) - np.min(predictions),
        'unique_count': len(np.unique(np.round(predictions, 10))),
        'zero_ratio': np.mean(predictions == 0),
        'is_constant': np.std(predictions) < 1e-6
    }
```

## Advanced Usage

### 1. Custom Configurations

#### 1.1 Creating Custom Config
```python
# config/custom_config.json
{
    "model": {
        "d_model": 128,           # Larger model
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.2
    },
    "training": {
        "batch_size": 128,        # Smaller batch
        "learning_rate": 0.0001,  # Higher learning rate
        "num_epochs": 20,         # Longer training
        "early_stopping_patience": 5
    }
}
```

#### 1.2 Using Custom Config
```bash
python train_transformer.py --config config/custom_config.json
```

### 2. Hyperparameter Tuning

#### 2.1 Grid Search Setup
```python
# hyperparameter_search.py
hyperparameters = {
    'd_model': [32, 64, 128],
    'n_heads': [2, 4, 8],
    'learning_rate': [0.00001, 0.00005, 0.0001],
    'dropout': [0.1, 0.2, 0.3]
}

for params in itertools.product(*hyperparameters.values()):
    config = create_config(params)
    results = train_model(config)
    save_results(params, results)
```

#### 2.2 Bayesian Optimization
```python
# Use optuna for advanced hyperparameter optimization
import optuna

def objective(trial):
    config = {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    ic = train_and_evaluate(config)
    return ic

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 3. Production Deployment

#### 3.1 Model Serving
```python
# model_server.py
import torch
from src.models import FuturesTransformer

class ModelServer:
    def __init__(self, model_path):
        self.model = FuturesTransformer(131)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, features, timestamps):
        with torch.no_grad():
            predictions = self.model(features, timestamps)
        return predictions.numpy()
```

#### 3.2 Real-time Inference
```python
# real_time_predictor.py
class RealTimePredictor:
    def __init__(self, model_path, scaler_path):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)
        self.buffer = SequenceBuffer(seq_len=20)
    
    def predict_next(self, new_data):
        # Update buffer with new data
        self.buffer.add(new_data)
        
        # Get sequence
        sequence = self.buffer.get_sequence()
        
        # Preprocess
        sequence_scaled = self.scaler.transform(sequence)
        
        # Predict
        prediction = self.model.predict(sequence_scaled)
        
        return prediction
```

## Best Practices

### 1. Data Quality
- **Validation**: Always validate data integrity before training
- **Preprocessing**: Use consistent preprocessing across train/val/test
- **Leakage**: Ensure no future information in features

### 2. Model Development
- **Baseline**: Start with simple models (FNN) before complex ones
- **Monitoring**: Track prediction diversity and gradient health
- **Validation**: Use proper time-series cross-validation

### 3. Experimentation
- **Reproducibility**: Set random seeds for consistent results
- **Logging**: Maintain detailed experiment logs
- **Comparison**: Always compare against baseline models

### 4. Performance Optimization
- **Batch Size**: Tune batch size for GPU memory efficiency
- **Mixed Precision**: Use automatic mixed precision for faster training
- **Data Loading**: Optimize data loading with proper num_workers

## Conclusion

This project guide provides a comprehensive framework for understanding and running the futures price prediction models. The key success factors are:

1. **Proper Data Processing**: Robust preprocessing and feature engineering
2. **Numerical Stability**: Careful attention to training stability
3. **Appropriate Evaluation**: Financial-specific metrics and validation
4. **Systematic Approach**: Structured experimentation and comparison

The project successfully demonstrates that deep learning can achieve meaningful predictive performance in financial markets when implemented with proper care and domain knowledge.

For questions or issues, please refer to the detailed documentation in the `docs/` directory or examine the training logs in the `experiments/` directory.
