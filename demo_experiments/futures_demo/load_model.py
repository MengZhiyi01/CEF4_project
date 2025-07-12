
# Model loading script
import pickle
import torch
import numpy as np
from src.models import FNN

# Load the model package
with open('demo_experiments/futures_demo/complete_model.pkl', 'rb') as f:
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
    """Make predictions and inverse transform if label scaler is available."""
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
print(f"Model info: {model.get_model_info()}")
print(f"Feature names: {len(feature_names)} features")
print(f"Label scaler available: {label_scaler is not None}")
print(f"Best validation loss: {model_package['best_metrics']['best_val_loss']:.6f}")
print(f"Best validation IC: {model_package['best_metrics']['best_val_ic']:.6f}")
