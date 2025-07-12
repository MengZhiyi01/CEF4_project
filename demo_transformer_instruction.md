# Transformer Demo Instructions

## Overview
This demo script provides a quick showcase of the Transformer model for futures price prediction, designed specifically for advisor presentations.

## Key Features
- **Quick Training**: Only 3 epochs for fast demonstration
- **Small Dataset**: Uses 50,000 samples for speed
- **Same Architecture**: Identical to production model
- **Visual Results**: Automatic generation of result plots
- **English Comments**: Clean, professional code

## Usage

### 1. Basic Demo Run
```bash
cd futures_prediction
python transformer_demo.py
```

### 2. Expected Output
The demo will show:
- Dataset initialization progress
- Model architecture details
- Training progress for 3 epochs
- Validation metrics after each epoch
- Final test evaluation
- Automatic visualization generation

### 3. Generated Files
- `demo_test_results.csv` - Detailed prediction results
- `transformer_demo_results.png` - 4-panel visualization

## Demo Results Interpretation

### Key Metrics to Highlight
1. **Information Coefficient (IC)**: Correlation between predictions and actual returns
2. **Prediction Diversity**: Number of unique predictions (shows model isn't stuck)
3. **RMSE**: Root Mean Square Error for prediction accuracy
4. **Visual Patterns**: Scatter plot shows prediction quality

### For Advisor Presentation
- **Training Speed**: ~2-3 minutes on GPU
- **Model Complexity**: 64-dim embeddings, 4 heads, 2 layers
- **Data Scale**: 131 features, 20-step sequences
- **Performance**: IC typically 0.02-0.05 (statistically significant)

## Technical Highlights

### Model Architecture
- **Time-Aware Transformer**: Uses temporal embeddings
- **Multi-Head Attention**: 4 attention heads for pattern recognition
- **Positional Encoding**: Handles sequence temporal structure
- **Gradient Clipping**: Ensures training stability

### Data Processing
- **Numerical Stability**: Handles NaN/Inf values
- **Outlier Removal**: 99th percentile clipping
- **Feature Scaling**: StandardScaler normalization
- **Sequence Creation**: 20-step lookback windows

### Training Stability
- **Adaptive Learning Rate**: Reduces on plateau
- **Gradient Clipping**: Prevents exploding gradients
- **Batch Processing**: Efficient GPU utilization
- **Early Monitoring**: Tracks prediction diversity

## Troubleshooting

### Common Issues
1. **"Data file not found"**: Ensure `final_filtered_data_1min.parquet` exists
2. **CUDA errors**: Script automatically falls back to CPU
3. **Memory issues**: Reduce demo_size in the script

### Performance Notes
- **GPU**: ~2-3 minutes total runtime
- **CPU**: ~10-15 minutes total runtime
- **Memory**: ~2GB RAM required

## Customization

### To Modify Demo Size
```python
# In transformer_demo.py, line ~420
demo_size = 50000  # Change this number
```

### To Extend Training
```python
# In transformer_demo.py, line ~450
n_epochs = 3  # Increase for longer training
```

## Expected Demo Flow

1. **Initialization** (30 seconds)
   - Load data
   - Create datasets
   - Initialize model

2. **Training** (1-2 minutes)
   - 3 epochs of training
   - Real-time metrics display
   - Validation after each epoch

3. **Evaluation** (30 seconds)
   - Test set evaluation
   - Metric calculation
   - Visualization generation

4. **Results** (immediate)
   - Performance summary
   - File generation
   - Success confirmation

## Key Points for Advisor Discussion

1. **Novel Architecture**: Time-aware attention mechanism
2. **Real Data**: Actual futures market data
3. **Practical Performance**: Statistically significant predictions
4. **Stability Solution**: Solved constant prediction problem
5. **Production Ready**: Same code as full training pipeline 