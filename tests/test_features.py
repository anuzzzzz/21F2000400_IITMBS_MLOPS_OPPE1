import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_preprocessing import create_features, load_and_preprocess_data

def test_rolling_average_feature():
    """Test rolling average calculation"""
    # Create sample data with 20 minutes
    data = {
        'timestamp': pd.date_range('2021-01-01 09:15', periods=20, freq='1T'),
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(110, 120, 20),
        'low': np.random.uniform(90, 100, 20),
        'close': [100 + i for i in range(20)],  # Ascending close prices
        'volume': np.random.randint(1000, 5000, 20)
    }
    df = pd.DataFrame(data)
    
    df_with_features = create_features(df)
    
    # Test rolling average exists and is calculated correctly
    assert 'rolling_avg_10' in df_with_features.columns
    assert df_with_features['rolling_avg_10'].dtype in ['float64', 'float32']
    assert not df_with_features['rolling_avg_10'].isna().all()
    
    # Test that rolling average is actually calculated correctly for 10th row
    if len(df_with_features) >= 10:
        expected_avg = df['close'][:10].mean()
        actual_avg = df_with_features.iloc[9]['rolling_avg_10']
        assert abs(expected_avg - actual_avg) < 0.001

def test_volume_sum_feature():
    """Test volume sum calculation"""
    data = {
        'timestamp': pd.date_range('2021-01-01 09:15', periods=20, freq='1T'),
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(110, 120, 20),
        'low': np.random.uniform(90, 100, 20),
        'close': np.random.uniform(95, 115, 20),
        'volume': [1000] * 20  # Fixed volume for easy testing
    }
    df = pd.DataFrame(data)
    
    df_with_features = create_features(df)
    
    # Test volume sum exists and is calculated correctly
    assert 'volume_sum_10' in df_with_features.columns
    assert df_with_features['volume_sum_10'].dtype in ['int64', 'float64']
    assert (df_with_features['volume_sum_10'] >= 0).all()
    
    # Test that volume sum is correct for 10th row
    if len(df_with_features) >= 10:
        expected_sum = 10000  # 10 * 1000
        actual_sum = df_with_features.iloc[9]['volume_sum_10']
        assert expected_sum == actual_sum

def test_target_creation():
    """Test target variable creation"""
    data = {
        'timestamp': pd.date_range('2021-01-01 09:15', periods=20, freq='1T'),
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(110, 120, 20),
        'low': np.random.uniform(90, 100, 20),
        'close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108, 92, 109, 91, 110],
        'volume': np.random.randint(1000, 5000, 20)
    }
    df = pd.DataFrame(data)
    
    df_with_features = create_features(df)
    
    # Test target exists and is binary
    assert 'target' in df_with_features.columns
    assert df_with_features['target'].dtype in ['int64', 'int32']
    assert set(df_with_features['target'].unique()).issubset({0, 1})
    
    # Test target calculation (first row: 100 vs 98 at +5 = 0)
    if len(df_with_features) > 0:
        # Close at t=0 is 100, close at t=5 is 103, so target should be 1
        expected_target = 1 if 103 > 100 else 0
        actual_target = df_with_features.iloc[0]['target']
        assert actual_target == expected_target

def test_data_sorting():
    """Test that data is properly sorted by timestamp"""
    # Create unsorted data
    timestamps = pd.date_range('2021-01-01 09:15', periods=10, freq='1T')
    shuffled_timestamps = timestamps.tolist()
    np.random.shuffle(shuffled_timestamps)
    
    data = {
        'timestamp': shuffled_timestamps,
        'open': np.random.uniform(100, 110, 10),
        'high': np.random.uniform(110, 120, 10),
        'low': np.random.uniform(90, 100, 10),
        'close': np.random.uniform(95, 115, 10),
        'volume': np.random.randint(1000, 5000, 10)
    }
    df = pd.DataFrame(data)
    
    df_with_features = create_features(df)
    
    # Test that timestamps are sorted
    assert df_with_features['timestamp'].is_monotonic_increasing

if __name__ == "__main__":
    test_rolling_average_feature()
    test_volume_sum_feature() 
    test_target_creation()
    test_data_sorting()
    print("All tests passed!")
