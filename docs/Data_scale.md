# Data Scaling Processing in Futures Price Prediction

## 1. Methodology

### 1.1 Theoretical Foundation

#### 1.1.1 Mathematical Principles of Label Scaling

Let the original labels be $y \in \mathbb{R}^n$, where $\mu_y = \frac{1}{n}\sum_{i=1}^n y_i$, $\sigma_y = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \mu_y)^2}$.

The standardization transformation is defined as:
$$\tilde{y}_i = \frac{y_i - \mu_y}{\sigma_y}$$

The transformed labels satisfy: $\mathbb{E}[\tilde{y}] = 0$, $\text{Var}[\tilde{y}] = 1$

#### 1.1.2 Gradient Propagation Optimization

For the mean squared error loss function:
$$L = \frac{1}{n}\sum_{i=1}^n (f(x_i) - y_i)^2$$

The gradient is:
$$\frac{\partial L}{\partial \theta} = \frac{2}{n}\sum_{i=1}^n (f(x_i) - y_i) \frac{\partial f(x_i)}{\partial \theta}$$

When $|y_i|$ is extremely small, the gradient term $(f(x_i) - y_i)$ is also of extremely small magnitude, leading to vanishing gradients. Through label scaling, gradient signals can be amplified to a reasonable range.

### 1.2 System Architecture Design

We designed an end-to-end data scaling processing architecture that includes the following core components:

1. **Label Scaler**: Responsible for label standardization and denormalization
2. **Dataset Enhancement**: Data loader integrated with scaling functionality
3. **Trainer Adaptation**: Training pipeline supporting scaled labels
4. **Evaluation Rescaling**: Ensuring correct calculation of evaluation metrics

## 2. Technical Implementation

### 2.1 Label Scaler Implementation

#### 2.1.1 Core Algorithm

```python
def _scale_labels(self):
    """Core algorithm for label scaling"""
    if self.label_scaler is None:
        self.label_scaler = StandardScaler()
    
    # Fit scaler and transform labels
    if self.fit_label_scaler:
        self.labels = self.label_scaler.fit_transform(
            self.labels.reshape(-1, 1)
        ).flatten()
    else:
        self.labels = self.label_scaler.transform(
            self.labels.reshape(-1, 1)
        ).flatten()
```

#### 2.1.2 Inverse Scaling Mechanism

```python
def inverse_transform_labels(self, scaled_labels: np.ndarray) -> np.ndarray:
    """Label inverse scaling"""
    if self.label_scaler is not None and self.scale_labels:
        return self.label_scaler.inverse_transform(
            scaled_labels.reshape(-1, 1)
        ).flatten()
    return scaled_labels
```

### 2.2 Dataset Architecture Enhancement

#### 2.2.1 Constructor Extension

Adding label scaling support to the `FuturesDataset` class:

```python
def __init__(self, 
             data: pd.DataFrame,
             feature_cols: List[str],
             label_col: str,
             # ... other parameters
             label_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
             fit_label_scaler: bool = False,
             scale_labels: bool = True):
```

#### 2.2.2 Scaling Process Integration

Label scaling is integrated into the data preprocessing pipeline:

```
Raw Data → Feature Engineering → Feature Standardization → Label Scaling → Tensor Conversion
```

### 2.3 Trainer Adaptation Mechanism

#### 2.3.1 Evaluation Function Modification

Implementing automatic inverse scaling during model evaluation:

```python
def evaluate(self, model: nn.Module, 
            data_loader: DataLoader,
            prefix: str = 'val',
            label_scaler: Optional[Any] = None):
    # ... forward propagation to get predictions
    
    # Inverse scaling for IC calculation
    if label_scaler is not None:
        original_predictions = label_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        original_labels = label_scaler.inverse_transform(
            labels.reshape(-1, 1)
        ).flatten()
    
    # Calculate IC using original scale
    overall_ic, _ = calculate_ic(original_labels, original_predictions)
```

#### 2.3.2 Enhanced Model Saving

Extending model saving functionality to include label scaler:

```python
model_package = {
    'model_state_dict': model.state_dict(),
    'model_config': self.config['model'],
    'scaler': scaler,
    'label_scaler': label_scaler,  # New addition
    'feature_names': feature_names,
    # ... other components
}
```

### 2.4 Data Flow Processing

#### 2.4.1 Training Data Flow

```
Original Labels → Standardization Scaling → Model Training → Scaled Predictions → Loss Calculation
```

#### 2.4.2 Evaluation Data Flow

```
Scaled Predictions → Inverse Scaling → Original Scale Predictions → IC Calculation → Performance Evaluation
```

## 3. Practical Application Guide

### 3.1 Usage Method

```bash
# Train model using label scaling configuration
python train.py --config config/config_label_scaled.json \
                --experiment_name futures_label_scaled
```

### 3.2 Configuration Parameters

Key configuration items:
- `scale_labels: true` - Enable label scaling
- `learning_rate: 0.001` - Adapted learning rate for scaled labels
- `optimizer: "adamw"` - Recommended optimizer

### 3.3 Model Loading and Prediction

```python
# Load complete model including scaler
model, scaler, label_scaler, feature_names = load_model()

# Automatic inverse scaling during prediction
predictions = predict_with_inverse_transform(
    model, scaler, label_scaler, feature_names, new_data
)
```