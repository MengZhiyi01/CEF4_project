"""
Sequence dataset for transformer-based futures prediction.
Handles time series data with proper temporal alignment and trading session awareness.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class FuturesSequenceDataset(Dataset):
    """High-performance dataset for sequence-based futures prediction with GPU acceleration."""
    
    def __init__(self,
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 label_col: str,
                 seq_len: int = 60,
                 pred_len: int = 1,
                 stride: int = 1,
                 scaler: Optional[Union[StandardScaler, RobustScaler, str]] = None,
                 fit_scaler: bool = False,
                 scale_label: bool = True,
                 add_time_features: bool = True,
                 normalize_by_instrument: bool = False,
                 cross_sectional_normalize: bool = False,
                 label_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
                 fit_label_scaler: bool = False,
                 device: str = 'cuda',
                 use_gpu_acceleration: bool = True):
        """
        Initialize the high-performance sequence dataset.
        
        Args:
            device: Device for GPU acceleration ('cuda' or 'cpu')
            use_gpu_acceleration: Whether to use GPU for data processing
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_gpu_acceleration = use_gpu_acceleration and torch.cuda.is_available()
        
        if self.use_gpu_acceleration:
            print(f"GPU acceleration enabled on {self.device}")
        else:
            print("Using CPU for data processing")
            
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.scale_label = scale_label
        self.add_time_features = add_time_features
        self.normalize_by_instrument = normalize_by_instrument
        self.cross_sectional_normalize = cross_sectional_normalize
        self.label_scaler = label_scaler
        self.fit_label_scaler = fit_label_scaler
        
        # Sort data by instrument and datetime
        self.data = self.data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
        
        # Convert datetime to pandas datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # Handle missing values
        self._handle_missing_values()
        
        # Add time features if requested
        if add_time_features:
            self._add_time_features()
            # Update feature columns to include time features
            time_feature_cols = [col for col in self.data.columns 
                               if col.startswith(('hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 
                                                'is_day_session', 'is_night_session', 'session_progress'))]
            self.feature_cols = self.feature_cols + time_feature_cols
            print(f"Added {len(time_feature_cols)} time features: {time_feature_cols}")
        
        # Extract hour for temporal features in sequences
        self.data['hour'] = self.data['datetime'].dt.hour
        
        # Process features and labels with GPU acceleration
        print(f"Feature processing summary:")
        print(f"  Original feature columns: {len(feature_cols)}")
        print(f"  Total columns after time features: {len(self.feature_cols)}")
        
        # Remove invalid feature columns
        invalid_features = [
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)"
        ]
        original_feature_count = len(self.feature_cols)
        self.feature_cols = [c for c in self.feature_cols if c not in invalid_features]
        removed_count = original_feature_count - len(self.feature_cols)
        if removed_count > 0:
            print(f"  Removed {removed_count} invalid feature columns")
        print(f"  Final feature count: {len(self.feature_cols)}")
        
        # Create feature matrix with GPU acceleration
        self._create_feature_matrix()
        
        # Handle outliers
        self._handle_outliers()
        
        # Initialize or use provided scaler
        if scaler is None:
            self.scaler = RobustScaler()
        else:
            self.scaler = scaler
            
        # Apply normalization with GPU acceleration
        if cross_sectional_normalize:
            self._normalize_cross_sectional_gpu()
        elif normalize_by_instrument:
            print("Warning: Per-instrument normalization not yet optimized for sequences")
            self._fit_global_scaler(fit_scaler)
        else:
            self._fit_global_scaler(fit_scaler)
            
        # Scale labels if needed
        if self.scale_label:
            self._scale_labels()
        
        # Create sequences for each instrument with optimization
        self.sequences = []
        self._create_sequences_optimized()
    
    def _create_feature_matrix(self):
        """Create feature matrix with optimized memory usage."""
        print("Creating optimized feature matrix...")
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.labels = self.data[self.label_col].values.astype(np.float32)
        
        # Pre-allocate memory to avoid fragmentation
        if self.use_gpu_acceleration:
            torch.cuda.empty_cache()  # Clear cache before processing
    
    def _handle_missing_values(self):
        """Handle missing values (same as FuturesDataset)"""
        # Forward fill for feature columns
        feature_data = self.data[self.feature_cols]
        feature_data = feature_data.ffill()
        feature_data = feature_data.bfill()
        feature_data = feature_data.fillna(0)
        self.data[self.feature_cols] = feature_data
        
        # Remove rows with missing labels
        label_missing = self.data[self.label_col].isna()
        if label_missing.sum() > 0:
            print(f"警告：删除了 {label_missing.sum()} 条标签缺失的数据")
            self.data = self.data[~label_missing]
    
    def _handle_outliers(self):
        """Handle outliers (same logic as FuturesDataset)"""
        self.features = np.asarray(self.features, dtype=np.float32)
        
        # Handle inf and nan values
        inf_mask = np.isinf(self.features) | np.isnan(self.features)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            print(f"警告：发现 {inf_count} 个无穷大或NaN值，将被替换")
            
            for col_idx in range(self.features.shape[1]):
                col_mask = inf_mask[:, col_idx]
                if np.any(col_mask):
                    valid_data = self.features[~col_mask, col_idx]
                    if len(valid_data) > 100:
                        p25, p75 = np.percentile(valid_data, [25, 75])
                        replacement_val = np.median(valid_data)
                        
                        pos_inf_mask = col_mask & np.isposinf(self.features[:, col_idx])
                        neg_inf_mask = col_mask & np.isneginf(self.features[:, col_idx])
                        nan_mask = col_mask & np.isnan(self.features[:, col_idx])
                        
                        if np.any(pos_inf_mask):
                            self.features[pos_inf_mask, col_idx] = p75
                        if np.any(neg_inf_mask):
                            self.features[neg_inf_mask, col_idx] = p25  
                        if np.any(nan_mask):
                            self.features[nan_mask, col_idx] = replacement_val
                    else:
                        self.features[col_mask, col_idx] = 0.0
        
        # Force replace any remaining inf values
        remaining_inf = np.isinf(self.features).sum()
        if remaining_inf > 0:
            print(f"警告：仍有 {remaining_inf} 个无穷大值，强制替换为0")
            self.features[np.isinf(self.features)] = 0.0
    
    def _add_time_features(self):
        """Add time features optimized for futures trading sessions."""
        # Convert datetime column to datetime type if not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # Extract time components
        hour = self.data['datetime'].dt.hour
        minute = self.data['datetime'].dt.minute
        
        # Cyclical encoding for hour and minute
        self.data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        self.data['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        self.data['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        
        # Futures trading session indicators with refined logic
        # Day session: 09:00-11:30, 13:30-15:00 (main trading hours)
        # Night session: 21:00-02:30 (night trading for most futures)
        day_morning = (hour >= 9) & (hour < 11) | ((hour == 11) & (minute <= 30))
        day_afternoon = (hour >= 13) & (hour < 15) | ((hour == 13) & (minute >= 30))
        day_mask = day_morning | day_afternoon
        
        night_mask = (hour >= 21) | (hour <= 2) | ((hour == 2) & (minute <= 30))
        
        self.data['is_day_session'] = day_mask.astype(float)
        self.data['is_night_session'] = night_mask.astype(float)
        
        # Session progress calculation with improved granularity
        session_progress = self._calculate_session_progress(hour, minute, day_mask, night_mask)
        self.data['session_progress'] = session_progress
        
        print(f"Time features added for futures trading sessions:")
    
    def _calculate_session_progress(self, hour: pd.Series, minute: pd.Series, 
                                  day_mask: pd.Series, night_mask: pd.Series) -> pd.Series:
        """
        Calculate session progress for futures trading.
        
        Day session: 09:00-11:30 (2.5h), 13:30-15:00 (1.5h) = 4h total
        Night session: 21:00-02:30 (5.5h)
        """
        progress = pd.Series(0.0, index=hour.index)
        
        # Day session progress
        day_indices = day_mask[day_mask].index
        for idx in day_indices:
            h, m = hour.iloc[idx], minute.iloc[idx]
            if 9 <= h < 11 or (h == 11 and m <= 30):
                # Morning session: 09:00-11:30 (2.5 hours)
                minutes_from_start = (h - 9) * 60 + m
                if h == 11 and m > 30:
                    minutes_from_start = 2.5 * 60  # Cap at session end
                progress.iloc[idx] = min(minutes_from_start / (2.5 * 60), 0.625)  # 0-0.625
            elif (h == 13 and m >= 30) or h == 14 or (h == 15 and m == 0):
                # Afternoon session: 13:30-15:00 (1.5 hours)
                minutes_from_start = (h - 13) * 60 + (m - 30)
                if h == 13 and m < 30:
                    minutes_from_start = 0
                progress.iloc[idx] = 0.625 + min(minutes_from_start / (1.5 * 60), 0.375)  # 0.625-1.0
        
        # Night session progress  
        night_indices = night_mask[night_mask].index
        for idx in night_indices:
            h, m = hour.iloc[idx], minute.iloc[idx]
            if h >= 21:
                # Night start: 21:00-23:59
                minutes_from_start = (h - 21) * 60 + m
                progress.iloc[idx] = minutes_from_start / (5.5 * 60)  # 5.5 hours total
            elif h <= 2 or (h == 2 and m <= 30):
                # Night continuation: 00:00-02:30
                if h <= 2:
                    minutes_from_start = (h + 3) * 60 + m  # +3 to continue from 21:00
                    if h == 2 and m > 30:
                        minutes_from_start = 5.5 * 60  # Cap at session end
                    progress.iloc[idx] = min(minutes_from_start / (5.5 * 60), 1.0)
        
        return progress
    
    def _normalize_cross_sectional_gpu(self):
        """Cross-sectional normalization for sequence data using GPU acceleration."""
        print("Applying cross-sectional normalization with GPU acceleration...")
        
        self.features = np.asarray(self.features, dtype=np.float32)
        
        # Get non-time feature columns for normalization
        non_time_feature_cols = [col for col in self.feature_cols 
                               if not col.startswith(('hour_', 'minute_', 'is_', 'session_', 'instrument_'))]
        
        if not non_time_feature_cols:
            print("Warning: No features found for cross-sectional normalization")
            self.scaler = "cross_sectional"
            return
        
        # Get feature indices for normalization
        feature_indices = np.array([self.feature_cols.index(col) for col in non_time_feature_cols])
        
        if self.use_gpu_acceleration:
            # GPU accelerated cross-sectional normalization
            self._gpu_cross_sectional_normalize(feature_indices)
        else:
            # CPU fallback
            self._cpu_cross_sectional_normalize(feature_indices)
        
        print(f"Cross-sectional normalization completed with GPU acceleration")
        print(f"  Normalized features: {len(feature_indices)}")
        
        # Set identifier for cross-sectional normalization
        self.scaler = "cross_sectional"
    
    def _gpu_cross_sectional_normalize(self, feature_indices):
        """GPU accelerated cross-sectional normalization."""
        # Convert datetime to tensor for efficient grouping
        datetime_values = pd.to_datetime(self.data['datetime']).values
        unique_datetimes = np.unique(datetime_values)
        
        # Process in batches to manage GPU memory
        batch_size = min(1000, len(unique_datetimes))
        standardized_count = 0
        
        for i in range(0, len(unique_datetimes), batch_size):
            batch_datetimes = unique_datetimes[i:i+batch_size]
            
            # Process batch on GPU
            batch_count = self._process_datetime_batch_gpu(batch_datetimes, feature_indices)
            standardized_count += batch_count
            
            # Clear GPU cache periodically
            if i % 10000 == 0 and self.use_gpu_acceleration:
                torch.cuda.empty_cache()
        
        print(f"  Processed time points: {standardized_count}/{len(unique_datetimes)}")
    
    def _process_datetime_batch_gpu(self, batch_datetimes, feature_indices):
        """Process a batch of datetimes on GPU."""
        count = 0
        
        for dt in batch_datetimes:
            dt_mask = self.data['datetime'] == dt
            dt_indices = np.where(dt_mask)[0]
            
            if len(dt_indices) < 2:
                continue
            
            # Move to GPU for computation
            dt_features = self.features[dt_indices][:, feature_indices]
            
            if self.use_gpu_acceleration:
                # GPU computation
                dt_features_gpu = torch.from_numpy(dt_features).to(self.device)
                
                # Calculate mean and std on GPU
                dt_mean = torch.mean(dt_features_gpu, dim=0)
                dt_std = torch.std(dt_features_gpu, dim=0)
                
                # Avoid division by zero
                dt_std = torch.where(dt_std < 1e-8, torch.ones_like(dt_std), dt_std)
                
                # Standardize on GPU
                standardized_features = (dt_features_gpu - dt_mean) / dt_std
                
                # Check for anomalies
                if torch.any(torch.isnan(standardized_features)) or torch.any(torch.isinf(standardized_features)):
                    continue
                
                # Move back to CPU and update
                self.features[dt_indices[:, None], feature_indices] = standardized_features.cpu().numpy()
            else:
                # CPU fallback
                dt_mean = np.mean(dt_features, axis=0)
                dt_std = np.std(dt_features, axis=0)
                dt_std = np.where(dt_std < 1e-8, 1.0, dt_std)
                standardized_features = (dt_features - dt_mean) / dt_std
                
                if np.any(np.isnan(standardized_features)) or np.any(np.isinf(standardized_features)):
                    continue
                
                self.features[dt_indices[:, None], feature_indices] = standardized_features
            
            count += 1
        
        return count
    
    def _cpu_cross_sectional_normalize(self, feature_indices):
        """CPU fallback for cross-sectional normalization."""
        unique_datetimes = self.data['datetime'].unique()
        standardized_count = 0
        
        for dt in unique_datetimes:
            dt_mask = self.data['datetime'] == dt
            dt_indices = np.where(dt_mask)[0]
            
            if len(dt_indices) < 2:
                continue
                
            # Extract features for this time point
            dt_features = self.features[dt_indices][:, feature_indices]
            
            # Calculate cross-sectional mean and std
            dt_mean = np.mean(dt_features, axis=0)
            dt_std = np.std(dt_features, axis=0)
            
            # Avoid division by zero
            dt_std = np.where(dt_std < 1e-8, 1.0, dt_std)
            
            # Apply cross-sectional standardization
            standardized_features = (dt_features - dt_mean) / dt_std
            
            # Validate results
            if np.any(np.isnan(standardized_features)) or np.any(np.isinf(standardized_features)):
                continue
            
            # Update feature matrix
            self.features[dt_indices[:, None], feature_indices] = standardized_features
            standardized_count += 1
            
        print(f"  Processed time points: {standardized_count}/{len(unique_datetimes)}")
    
    def _fit_global_scaler(self, fit_scaler: bool):
        """Fit global scaler"""
        if isinstance(self.scaler, str):
            # Cross-sectional normalization already applied
            return
            
        if fit_scaler:
            # Clean infinite values before fitting
            all_features = np.where(np.isfinite(self.features), self.features, 0.0)
            if not np.isfinite(all_features).all():
                all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.scaler.fit(all_features)
    
    def _scale_labels(self):
        """Scale labels for model training."""
        if self.label_scaler is None:
            self.label_scaler = StandardScaler()
        
        # Check original label statistics
        original_mean = self.labels.mean()
        original_std = self.labels.std()
        
        print(f"Label statistics:")
        print(f"  Original - mean: {original_mean:.8f}, std: {original_std:.8f}")
        
        # Apply label scaling
        if self.fit_label_scaler:
            self.labels = self.label_scaler.fit_transform(self.labels.reshape(-1, 1)).flatten()
            print(f"  Label scaler fitted with scale factor: {self.label_scaler.scale_[0]:.8f}")
        else:
            self.labels = self.label_scaler.transform(self.labels.reshape(-1, 1)).flatten()
        
        # Check scaled label statistics
        scaled_mean = self.labels.mean()
        scaled_std = self.labels.std()
        
        print(f"  Scaled - mean: {scaled_mean:.6f}, std: {scaled_std:.6f}")
    
    def _create_sequences_optimized(self):
        """Create sequences for each instrument with optimized memory usage and parallel processing."""
        instruments = self.data['instrument'].unique()
        print(f"Creating sequences for {len(instruments)} instruments...")
        
        # Pre-compute instrument data indices for faster access
        instrument_indices = {}
        for instrument in instruments:
            instrument_indices[instrument] = self.data[self.data['instrument'] == instrument].index.values
        
        total_sequences = 0
        
        # Process instruments in batches
        for idx, instrument in enumerate(instruments):
            if idx % 10 == 0:
                print(f"Processing instrument {idx+1}/{len(instruments)}: {instrument}")
            
            inst_indices = instrument_indices[instrument]
            inst_data_subset = self.data.iloc[inst_indices]
            
            # Skip if not enough data
            if len(inst_data_subset) < self.seq_len + self.pred_len:
                continue
            
            # Optimized sequence creation with vectorized operations
            sequences_count = self._create_sequences_for_instrument_vectorized(
                inst_data_subset, inst_indices, instrument
            )
            total_sequences += sequences_count
            
            # Periodic memory cleanup
            if idx % 20 == 0 and self.use_gpu_acceleration:
                torch.cuda.empty_cache()
            
        print(f"Created {total_sequences} sequences in total.")
    
    def _create_sequences_for_instrument_vectorized(self, inst_data, inst_indices, instrument):
        """Vectorized sequence creation for a single instrument."""
        sequences_count = 0
        
        # Pre-compute time series for efficiency
        timestamps = inst_data['datetime'].values
        
        # Vectorized approach for time continuity check
        valid_starts = []
        
        for i in range(0, len(inst_data) - self.seq_len - self.pred_len + 1, self.stride):
            # Quick time continuity check
            start_time = timestamps[i]
            end_time = timestamps[i + self.seq_len + self.pred_len - 1]
            
            try:
                # Use numpy datetime64 for faster computation
                time_diff = np.datetime64(end_time) - np.datetime64(start_time)
                if time_diff <= np.timedelta64(2, 'h'):  # 2 hour threshold
                    valid_starts.append(i)
            except:
                continue
                
        # Batch create sequence info
        for i in valid_starts:
            sequence_info = {
                'instrument': instrument,
                'start_idx': inst_indices[i],
                'end_idx': inst_indices[i + self.seq_len - 1],
                'label_idx': inst_indices[i + self.seq_len + self.pred_len - 1],
                'start_time': inst_data.iloc[i]['datetime'],
                'end_time': inst_data.iloc[i + self.seq_len - 1]['datetime'],
                'label_time': inst_data.iloc[i + self.seq_len + self.pred_len - 1]['datetime']
            }
            
            self.sequences.append(sequence_info)
            sequences_count += 1
            
        return sequences_count
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sequence sample.
        
        Returns:
            features: Tensor of shape (seq_len, feature_dim)
            label: Tensor of shape (1,) or (pred_len,)
            timestamps: Tensor of shape (seq_len,) containing hour values
            metadata: Dictionary with additional information
        """
        seq_info = self.sequences[idx]
        
        # Extract sequence data
        start_idx = seq_info['start_idx']
        end_idx = seq_info['end_idx']
        label_idx = seq_info['label_idx']
        
        # Get features
        features = self.data.loc[start_idx:end_idx, self.feature_cols].values
        
        # Get timestamps (hours)
        timestamps = self.data.loc[start_idx:end_idx, 'hour'].values
        
        # Get label
        label = self.data.loc[label_idx, self.label_col]
        
        # Scale features (skip if cross-sectional normalization already applied)
        if isinstance(self.scaler, str) and self.scaler == "cross_sectional":
            # Features already normalized during initialization
            pass
        else:
            # Apply scaler transformation
            features = self.scaler.transform(features)
        
        # Scale label if needed
        if self.scale_label and hasattr(self, 'label_scaler'):
            label = self.label_scaler.transform([[label]])[0, 0]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.FloatTensor([label])
        timestamps = torch.LongTensor(timestamps)
        
        # Metadata (convert timestamps to strings to avoid collate issues)
        metadata = {
            'instrument': seq_info['instrument'],
            'start_time': str(seq_info['start_time']),
            'end_time': str(seq_info['end_time']),
            'label_time': str(seq_info['label_time'])
        }
        
        return features, label, timestamps, metadata
    
    def get_feature_names(self) -> List[str]:
        """Get feature column names."""
        return self.feature_cols
    
    def get_scaler(self):
        """Get the feature scaler."""
        return self.scaler
    
    def get_label_scaler(self):
        """Get the label scaler."""
        return self.label_scaler if hasattr(self, 'label_scaler') else None


def create_sequence_dataloaders(train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               feature_cols: List[str],
                               label_col: str,
                               seq_len: int = 60,
                               pred_len: int = 1,
                               batch_size: int = 32,
                               num_workers: int = 4,
                               scale_label: bool = True,
                               add_time_features: bool = True,
                               normalize_by_instrument: bool = False,
                               cross_sectional_normalize: bool = False,
                               device: str = 'cuda',
                               use_gpu_acceleration: bool = True) -> Dict[str, Union[torch.utils.data.DataLoader, Any]]:
    """
    Create high-performance data loaders for sequence data with GPU acceleration.
    
    Args:
        device: Device for GPU acceleration
        use_gpu_acceleration: Whether to use GPU for data processing
        
    Returns:
        Dictionary with train, val, test data loaders, scaler, and label_scaler
    """
    print(f"Creating high-performance sequence data loaders with cross_sectional_normalize={cross_sectional_normalize}")
    
    # Optimize num_workers based on system
    import os
    optimal_workers = min(num_workers, os.cpu_count() or 1)
    
    # Create datasets with GPU acceleration
    train_dataset = FuturesSequenceDataset(
        train_data, feature_cols, label_col,
        seq_len=seq_len, pred_len=pred_len,
        fit_scaler=True, scale_label=scale_label,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        fit_label_scaler=True,
        device=device,
        use_gpu_acceleration=use_gpu_acceleration
    )
    
    # Get the scaler from training dataset
    train_scaler = train_dataset.get_scaler()
    train_label_scaler = train_dataset.get_label_scaler()
    
    # Use the same scaler for validation and test
    val_dataset = FuturesSequenceDataset(
        val_data, feature_cols, label_col,
        seq_len=seq_len, pred_len=pred_len,
        scaler=train_scaler,
        fit_scaler=False, scale_label=scale_label,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        label_scaler=train_label_scaler,
        fit_label_scaler=False,
        device=device,
        use_gpu_acceleration=use_gpu_acceleration
    )
    
    test_dataset = FuturesSequenceDataset(
        test_data, feature_cols, label_col,
        seq_len=seq_len, pred_len=pred_len,
        scaler=train_scaler,
        fit_scaler=False, scale_label=scale_label,
        add_time_features=add_time_features,
        normalize_by_instrument=normalize_by_instrument,
        cross_sectional_normalize=cross_sectional_normalize,
        label_scaler=train_label_scaler,
        fit_label_scaler=False,
        device=device,
        use_gpu_acceleration=use_gpu_acceleration
    )
    
    # Create high-performance data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,
        persistent_workers=True if optimal_workers > 0 else False,  # Keep workers alive
        prefetch_factor=4 if optimal_workers > 0 else 2  # Prefetch batches
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=4 if optimal_workers > 0 else 2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=4 if optimal_workers > 0 else 2
    )
    
    print(f"High-performance sequence dataloaders created:")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} sequences")
    print(f"  Val: {len(val_loader)} batches, {len(val_dataset)} sequences") 
    print(f"  Test: {len(test_loader)} batches, {len(test_dataset)} sequences")
    print(f"  Feature dim: {len(train_dataset.feature_cols)}")
    print(f"  Optimizations: pin_memory=True, num_workers={optimal_workers}, prefetch_factor=4")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'scaler': train_scaler,
        'label_scaler': train_label_scaler
    } 