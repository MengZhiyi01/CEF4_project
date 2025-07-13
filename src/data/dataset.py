"""
期货数据集类
处理期货数据的加载、预处理和特征工程
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict, Union
from sklearn.preprocessing import StandardScaler, RobustScaler


class FuturesDataset(Dataset):
    """期货预测数据集类"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 label_col: str,
                 scaler: Optional[Union[Dict[str, Union[StandardScaler, RobustScaler]], Union[StandardScaler, RobustScaler]]] = None,
                 fit_scaler: bool = False,
                 add_time_features: bool = True,
                 normalize_by_instrument: bool = True,
                 cross_sectional_normalize: bool = False,
                 label_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
                 fit_label_scaler: bool = False,
                 scale_labels: bool = True):
        """
        Args:
            data: 原始数据DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            scaler: 特征标准化器字典（按品种），如果为None则创建新的
            fit_scaler: 是否拟合scaler（训练集为True，验证/测试集为False）
            add_time_features: 是否添加时间特征
            normalize_by_instrument: 是否按品种分别标准化
            cross_sectional_normalize: 是否使用截面标准化（在每个时间点对所有资产标准化）
            label_scaler: 标签缩放器
            fit_label_scaler: 是否拟合标签缩放器
            scale_labels: 是否缩放标签
        """
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.add_time_features = add_time_features
        self.normalize_by_instrument = normalize_by_instrument
        self.cross_sectional_normalize = cross_sectional_normalize
        self.scale_labels = scale_labels
        self.label_scaler = label_scaler
        self.fit_label_scaler = fit_label_scaler
        
        # 处理缺失值
        self._handle_missing_values()
        
        # 添加时间特征
        if add_time_features:
            self._add_time_features()
            # 更新特征列
            time_feature_cols = [col for col in self.data.columns 
                               if col.startswith(('hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 
                                                'is_day_session', 'is_night_session', 'session_progress'))]
            self.feature_cols = self.feature_cols + time_feature_cols
        
        # 添加品种编码特征
        self._add_instrument_features()
        
        # 提取特征和标签
        print(f"提取特征和标签 - 特征列数: {len(self.feature_cols)}")
        # 移除已知含无穷大值的相关性特征列
        invalid_features = [
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
            "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)"
        ]
        original_feature_count = len(self.feature_cols)
        self.feature_cols = [c for c in self.feature_cols if c not in invalid_features]
        removed_count = original_feature_count - len(self.feature_cols)
        if removed_count > 0:
            print(f"已移除 {removed_count} 个无效特征列: {invalid_features}")
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.labels = self.data[self.label_col].values.astype(np.float32)
        
        # 首先检查原始数据状态
        inf_count = np.isinf(self.features).sum()
        nan_count = np.isnan(self.features).sum()
        print(f"原始特征数据: 无穷大值={inf_count}, NaN值={nan_count}")
        
        # 处理异常值（无穷大、过大值等）- 必须在标准化之前
        self._handle_outliers()
        
        # 验证异常值处理效果
        inf_count_after = np.isinf(self.features).sum()
        nan_count_after = np.isnan(self.features).sum()
        print(f"异常值处理后: 无穷大值={inf_count_after}, NaN值={nan_count_after}")
        print(f"特征范围: [{self.features.min():.4f}, {self.features.max():.4f}]")
        
        # 标准化特征
        if cross_sectional_normalize:
            self._normalize_cross_sectional()
        elif normalize_by_instrument:
            self._normalize_by_instrument(scaler, fit_scaler)
        else:
            # 如果是字典类型的scaler，提取单个scaler或创建新的
            if isinstance(scaler, dict):
                single_scaler = None  # 字典类型不适用于全局标准化
            else:
                single_scaler = scaler
            self._normalize_globally(single_scaler, fit_scaler)
            
        # 验证标准化效果
        print(f"标准化后特征统计: mean={self.features.mean():.4f}, std={self.features.std():.4f}")
        print(f"标准化后范围: [{self.features.min():.4f}, {self.features.max():.4f}]")
        
        # 标签缩放
        if self.scale_labels:
            self._scale_labels()
        
        # 转换为张量
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)
        
        # 保存元数据
        self.instruments = self.data['instrument'].values
        self.datetimes = self.data['datetime'].values
        
    def _handle_missing_values(self):
        """处理缺失值"""
        # 对特征列进行前向填充
        feature_data = self.data[self.feature_cols]
        feature_data = feature_data.ffill()  # type: ignore
        # 对剩余的缺失值使用后向填充
        feature_data = feature_data.bfill()  # type: ignore
        # 如果还有缺失值，使用0填充
        feature_data = feature_data.fillna(0)
        self.data[self.feature_cols] = feature_data
        
        # 对标签列，删除缺失值
        label_missing = self.data[self.label_col].isna()  # type: ignore
        if label_missing.sum() > 0:  # type: ignore
            print(f"警告：删除了 {label_missing.sum()} 条标签缺失的数据")
            self.data = self.data[~label_missing]
    
    def _handle_outliers(self):
        """处理异常值（无穷大、过大值等）"""
        # 确保features是numpy数组
        self.features = np.asarray(self.features, dtype=np.float32)
        
        # 1. 首先处理无穷大和NaN值 - 在标准化之前处理
        inf_mask = np.isinf(self.features) | np.isnan(self.features)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            print(f"警告：发现 {inf_count} 个无穷大或NaN值，将被替换")
            
            # 按列处理，用更稳健的方法替换
            for col_idx in range(self.features.shape[1]):
                col_mask = inf_mask[:, col_idx]
                if np.any(col_mask):
                    valid_data = self.features[~col_mask, col_idx]
                    if len(valid_data) > 100:  # 需要足够的有效数据
                        # 使用分位数来替换，更稳健
                        p25, p75 = np.percentile(valid_data, [25, 75])
                        replacement_val = np.median(valid_data)  # 使用中位数
                        
                        # 对于正无穷使用75分位数，负无穷使用25分位数
                        pos_inf_mask = col_mask & np.isposinf(self.features[:, col_idx])
                        neg_inf_mask = col_mask & np.isneginf(self.features[:, col_idx])
                        nan_mask = col_mask & np.isnan(self.features[:, col_idx])
                        
                        if np.any(pos_inf_mask):
                            self.features[pos_inf_mask, col_idx] = p75
                        if np.any(neg_inf_mask):
                            self.features[neg_inf_mask, col_idx] = p25  
                        if np.any(nan_mask):
                            self.features[nan_mask, col_idx] = replacement_val
                            
                        print(f"  列 {col_idx}: 替换了 {np.sum(col_mask)} 个异常值")
                    else:
                        # 如果有效数据太少，使用0
                        self.features[col_mask, col_idx] = 0.0
                        print(f"  列 {col_idx}: 有效数据不足，使用0替换")
        
        # 1.5 再次检查是否还有无穷大值
        remaining_inf = np.isinf(self.features).sum()
        if remaining_inf > 0:
            print(f"警告：仍有 {remaining_inf} 个无穷大值，强制替换为0")
            self.features[np.isinf(self.features)] = 0.0
        
        # 2. 使用更稳健的异常值检测方法 - 基于MAD（中位数绝对偏差）
        for col_idx in range(self.features.shape[1]):
            col_data = self.features[:, col_idx]
            valid_data = col_data[np.isfinite(col_data)]
            
            if len(valid_data) > 0:
                # 计算MAD
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                
                # 如果MAD太小，使用标准差的稳健估计
                if mad < 1e-8:
                    mad = 1.4826 * np.percentile(np.abs(valid_data - median), 75)
                
                # 定义异常值边界（使用5倍MAD）
                if mad > 1e-8:
                    lower_bound = median - 5 * mad
                    upper_bound = median + 5 * mad
                    
                    # 限制极端值
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    if np.any(outlier_mask):
                        # 使用边界值替换，而不是删除
                        col_data[col_data < lower_bound] = lower_bound
                        col_data[col_data > upper_bound] = upper_bound
                        self.features[:, col_idx] = col_data
        
        # 3. 处理标签中的异常值
        if np.any(np.isinf(self.labels) | np.isnan(self.labels)):
            label_inf_mask = np.isinf(self.labels) | np.isnan(self.labels)
            label_inf_count = np.sum(label_inf_mask)
            print(f"警告：发现 {label_inf_count} 个标签异常值，对应的样本将被删除")
            
            # 删除标签异常的样本
            valid_mask = ~label_inf_mask
            self.features = self.features[valid_mask]
            self.labels = self.labels[valid_mask]
            
            # 同时更新数据框
            self.data = self.data.iloc[valid_mask].reset_index(drop=True)
        
        # 4. 最终检查：确保没有剩余的异常值
        final_check = np.isinf(self.features).sum() + np.isnan(self.features).sum()
        if final_check > 0:
            print(f"错误：处理后仍有 {final_check} 个异常值！")
            # 强制替换为0
            self.features[np.isinf(self.features) | np.isnan(self.features)] = 0
    
    def _add_time_features(self):
        """添加时间特征，考虑期货日盘夜盘特点"""
        # 确保datetime列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
        # 提取基础时间特征
        hour = self.data['datetime'].dt.hour
        minute = self.data['datetime'].dt.minute
        day_of_week = self.data['datetime'].dt.dayofweek
        
        # 创建新特征字典，一次性添加所有列
        new_features = {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'minute_sin': np.sin(2 * np.pi * minute / 60),
            'minute_cos': np.cos(2 * np.pi * minute / 60),
        }
        
        # 判断是否为日盘或夜盘时段
        # 日盘时段：9:00-10:15, 10:30-11:30, 13:30-15:00
        day_mask = (
            ((hour == 9) | 
             (hour == 10) & (minute <= 15)) |
            ((hour == 10) & (minute >= 30) |
             (hour == 11) & (minute <= 30)) |
            ((hour == 13) & (minute >= 30) |
             (hour == 14) | 
             (hour == 15) & (minute == 0))
        )
        
        # 夜盘时段：21:00-23:59, 00:00-02:30
        night_mask = (
            (hour >= 21) |  # 21:00-23:59
            (hour <= 2) |   # 00:00-02:00
            ((hour == 2) & (minute <= 30))  # 02:00-02:30
        )
        
        new_features['is_day_session'] = day_mask.astype(int)
        new_features['is_night_session'] = night_mask.astype(int)
        
        # 计算交易时段进度
        session_progress = self._calculate_session_progress_vectorized(hour, minute, day_mask, night_mask)
        new_features['session_progress'] = session_progress
        
        # 一次性添加所有新特征
        time_features_df = pd.DataFrame(new_features, index=self.data.index)
        self.data = pd.concat([self.data, time_features_df], axis=1)
        
    def _calculate_session_progress_vectorized(self, hour: pd.Series, minute: pd.Series, 
                                              day_mask: pd.Series, night_mask: pd.Series) -> pd.Series:
        """计算在交易时段内的进度（0-1）- 向量化版本"""
        progress = pd.Series(0.5, index=self.data.index)  # 默认值
        total_minutes = hour * 60 + minute
        
        # 日盘时段进度计算
        day_indices = day_mask[day_mask].index  # type: ignore
        for idx in day_indices:
            tm = total_minutes.loc[idx]
            if 9 * 60 <= tm <= 10 * 60 + 15:  # 9:00-10:15
                progress.loc[idx] = (tm - 9 * 60) / 75
            elif 10 * 60 + 30 <= tm <= 11 * 60 + 30:  # 10:30-11:30
                progress.loc[idx] = (tm - 10 * 60 - 30) / 60
            elif 13 * 60 + 30 <= tm <= 15 * 60:  # 13:30-15:00
                progress.loc[idx] = (tm - 13 * 60 - 30) / 90
                
        # 夜盘时段进度计算
        night_indices = night_mask[night_mask].index  # type: ignore
        for idx in night_indices:
            h = hour.loc[idx]
            tm = total_minutes.loc[idx]
            if h >= 21:  # 21:00-23:59
                progress.loc[idx] = (tm - 21 * 60) / 180
            else:  # 00:00-02:30
                progress.loc[idx] = (tm + 180) / 330
                
        return progress
    
    def _calculate_session_progress(self) -> pd.Series:
        """计算在交易时段内的进度（0-1）"""
        progress = pd.Series(0.5, index=self.data.index)  # 默认值
        
        # 日盘时段进度计算
        for idx, row in self.data.iterrows():
            hour, minute = row['hour'], row['minute']
            total_minutes = hour * 60 + minute
            
            if row['is_day_session']:  # type: ignore
                if 9 * 60 <= total_minutes <= 10 * 60 + 15:  # 9:00-10:15
                    progress.loc[idx] = (total_minutes - 9 * 60) / 75
                elif 10 * 60 + 30 <= total_minutes <= 11 * 60 + 30:  # 10:30-11:30
                    progress.loc[idx] = (total_minutes - 10 * 60 - 30) / 60
                elif 13 * 60 + 30 <= total_minutes <= 15 * 60:  # 13:30-15:00
                    progress.loc[idx] = (total_minutes - 13 * 60 - 30) / 90
                    
            elif row['is_night_session']:  # type: ignore
                if hour >= 21:  # 21:00-23:59
                    progress.loc[idx] = (total_minutes - 21 * 60) / 180
                else:  # 00:00-02:30
                    progress.loc[idx] = (total_minutes + 180) / 330
                    
        return progress
    
    def _add_instrument_features(self):
        """添加品种编码特征"""
        # 获取唯一品种列表
        unique_instruments = sorted(self.data['instrument'].unique())
        self.instrument_to_id = {inst: i for i, inst in enumerate(unique_instruments)}
        self.id_to_instrument = {i: inst for inst, i in self.instrument_to_id.items()}
        
        # 添加品种ID
        self.data['instrument_id'] = self.data['instrument'].map(lambda x: self.instrument_to_id[x])
        
        # 添加品种类型编码（基于品种名称的前缀）
        self.data['instrument_type'] = self.data['instrument'].str.extract(r'([A-Z]+)')[0]
        unique_types = sorted(self.data['instrument_type'].unique())
        type_to_id = {type_: i for i, type_ in enumerate(unique_types)}
        self.data['instrument_type_id'] = self.data['instrument_type'].map(lambda x: type_to_id[x])
        
        # 添加到特征列
        self.feature_cols.extend(['instrument_id', 'instrument_type_id'])
        
        print(f"添加品种特征: {len(unique_instruments)} 个品种, {len(unique_types)} 个品种类型")
    
    def _normalize_by_instrument(self, scaler: Optional[Union[Dict[str, Union[StandardScaler, RobustScaler]], Union[StandardScaler, RobustScaler]]], fit_scaler: bool):
        """按品种分别标准化特征"""
        if scaler is None:
            self.scaler = {}
        elif isinstance(scaler, dict):
            self.scaler = scaler
        else:
            # 如果传入的是单个scaler，转换为全局标准化
            print("警告：传入单个scaler，将使用全局标准化")
            self._normalize_globally(scaler, fit_scaler)
            return
            
        # 获取非品种相关的特征列（时间特征等）
        global_feature_cols = [col for col in self.feature_cols 
                              if col.startswith(('hour_', 'minute_', 'is_', 'session_'))]
        instrument_feature_cols = [col for col in self.feature_cols 
                                  if col not in global_feature_cols and not col.startswith('instrument_')]
        
        normalized_features = np.copy(self.features)
        
        # 1. 全局标准化时间特征
        if global_feature_cols:
            global_indices = np.array([self.feature_cols.index(col) for col in global_feature_cols])
            global_scaler_key = '_global_'
            
            if fit_scaler:
                if global_scaler_key not in self.scaler:
                    self.scaler[global_scaler_key] = RobustScaler()
                normalized_features[:, global_indices] = self.scaler[global_scaler_key].fit_transform(
                    self.features[:, global_indices])
            else:
                if global_scaler_key in self.scaler:
                    normalized_features[:, global_indices] = self.scaler[global_scaler_key].transform(
                        self.features[:, global_indices])
        
        # 2. 按品种标准化其他特征
        if instrument_feature_cols:
            instrument_indices = np.array([self.feature_cols.index(col) for col in instrument_feature_cols])
            
            for instrument in self.data['instrument'].unique():
                instrument_mask = self.data['instrument'] == instrument
                instrument_data = self.features[instrument_mask][:, instrument_indices]
                
                if len(instrument_data) > 1:  # 至少需要2个样本
                    if fit_scaler:
                        if instrument not in self.scaler:
                            self.scaler[instrument] = RobustScaler()
                        normalized_instrument_data = self.scaler[instrument].fit_transform(instrument_data)
                    else:
                        if instrument in self.scaler:
                            normalized_instrument_data = self.scaler[instrument].transform(instrument_data)
                        else:
                            # 如果scaler中没有这个品种，使用原始数据
                            print(f"警告: 品种 {instrument} 的scaler不存在，使用原始数据")
                            normalized_instrument_data = instrument_data
                    
                    normalized_features[instrument_mask][:, instrument_indices] = normalized_instrument_data
                else:
                    print(f"警告: 品种 {instrument} 只有 {len(instrument_data)} 个样本，跳过标准化")
        
        # 3. 品种编码特征不需要标准化
        self.features = normalized_features
        
        print(f"按品种标准化完成: {len(self.scaler)} 个scaler (包括全局和各品种)")
    
    def _normalize_cross_sectional(self):
        """截面标准化 - 在每个时间点对所有资产的特征进行标准化"""
        print("开始截面标准化...")
        
        # 确保features是numpy数组
        self.features = np.asarray(self.features, dtype=np.float32)
        
        # 获取非时间和非品种特征的列索引
        non_time_feature_cols = [col for col in self.feature_cols 
                               if not col.startswith(('hour_', 'minute_', 'is_', 'session_', 'instrument_'))]
        
        if not non_time_feature_cols:
            print("警告：没有找到需要截面标准化的特征")
            return
        
        # 获取特征索引
        feature_indices = np.array([self.feature_cols.index(col) for col in non_time_feature_cols])
        
        # 按时间分组进行标准化
        unique_datetimes = self.data['datetime'].unique()
        standardized_count = 0
        
        for dt in unique_datetimes:
            dt_mask = self.data['datetime'] == dt
            dt_indices = np.where(dt_mask)[0]
            
            if len(dt_indices) < 2:  # 至少需要2个样本才能计算标准差
                continue
                
            # 提取该时间点的特征数据
            dt_features = self.features[dt_indices][:, feature_indices]
            
            # 计算均值和标准差
            dt_mean = np.mean(dt_features, axis=0)
            dt_std = np.std(dt_features, axis=0)
            
            # 避免除以0
            dt_std = np.where(dt_std < 1e-8, 1.0, dt_std)
            
            # 标准化
            standardized_features = (dt_features - dt_mean) / dt_std
            
            # 检查是否有异常值
            if np.any(np.isnan(standardized_features)) or np.any(np.isinf(standardized_features)):
                print(f"警告：时间点 {dt} 的标准化结果包含异常值，跳过")
                continue
            
            # 更新特征矩阵
            self.features[dt_indices[:, None], feature_indices] = standardized_features
            standardized_count += 1
            
        print(f"截面标准化完成: 处理了 {standardized_count}/{len(unique_datetimes)} 个时间点")
        print(f"截面标准化特征数: {len(feature_indices)}")
        
        # 验证标准化效果
        print(f"截面标准化后特征统计: mean={self.features[:, feature_indices].mean():.4f}, std={self.features[:, feature_indices].std():.4f}")
        
        # 设置标识，表明使用了截面标准化
        self.scaler = "cross_sectional"  # 设置为字符串标识而不是None
    
    def _normalize_globally(self, scaler: Optional[Union[StandardScaler, RobustScaler]], fit_scaler: bool):
        """全局标准化特征（原有方法）"""
        if scaler is None:
            self.scaler = RobustScaler()  # 对异常值更鲁棒
        else:
            self.scaler = scaler
            
        if fit_scaler:
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features = self.scaler.transform(self.features)
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        返回：
            features: 特征张量
            label: 标签张量
            metadata: 包含instrument和datetime的字典
        """
        metadata = {
            'instrument': str(self.instruments[idx]),
            'datetime': str(self.datetimes[idx])
        }
        return self.features[idx], self.labels[idx], metadata  # type: ignore
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_cols 
    
    def _scale_labels(self):
        """标签缩放"""
        if self.label_scaler is None:
            # 使用StandardScaler来缩放标签，这样会将标签的标准差变为1
            self.label_scaler = StandardScaler()
        
        # 检查原始标签统计
        original_mean = self.labels.mean()
        original_std = self.labels.std()
        original_range = [self.labels.min(), self.labels.max()]
        
        print(f"原始标签统计: mean={original_mean:.8f}, std={original_std:.8f}")
        print(f"原始标签范围: [{original_range[0]:.8f}, {original_range[1]:.8f}]")
        
        # 缩放标签
        if self.fit_label_scaler:
            # 训练集：拟合并转换
            self.labels = self.label_scaler.fit_transform(self.labels.reshape(-1, 1)).flatten()
            print(f"标签缩放器已拟合: scale_factor={self.label_scaler.scale_[0]:.8f}")
        else:
            # 验证/测试集：仅转换
            self.labels = self.label_scaler.transform(self.labels.reshape(-1, 1)).flatten()
        
        # 检查缩放后的标签统计
        scaled_mean = self.labels.mean()
        scaled_std = self.labels.std()
        scaled_range = [self.labels.min(), self.labels.max()]
        
        print(f"缩放后标签统计: mean={scaled_mean:.6f}, std={scaled_std:.6f}")
        print(f"缩放后标签范围: [{scaled_range[0]:.6f}, {scaled_range[1]:.6f}]")
        
        # 验证缩放是否成功
        if scaled_std < 0.1:
            print("警告：缩放后标签标准差仍然较小，可能需要调整缩放策略")
        elif scaled_std > 100:
            print("警告：缩放后标签标准差过大，可能需要调整缩放策略")
        else:
            print("✓ 标签缩放成功")
    
    def get_label_scaler(self) -> Optional[Union[StandardScaler, RobustScaler]]:
        """获取标签缩放器"""
        return self.label_scaler
    
    def inverse_transform_labels(self, scaled_labels: np.ndarray) -> np.ndarray:
        """反缩放标签"""
        if self.label_scaler is not None and self.scale_labels:
            return self.label_scaler.inverse_transform(scaled_labels.reshape(-1, 1)).flatten()
        return scaled_labels 