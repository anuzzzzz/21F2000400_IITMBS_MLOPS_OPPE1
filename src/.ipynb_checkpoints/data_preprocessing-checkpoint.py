import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

def load_and_preprocess_data(file_path):
    """Load CSV and handle missing minutes"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert timestamp and sort by time (CRITICAL: don't assume sorted)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create complete minute range for trading hours
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    # Generate all trading minutes (9:15 AM to 3:30 PM, Mon-Fri)
    all_minutes = pd.date_range(start=start_time, end=end_time, freq='1T')
    trading_minutes = []
    
    for minute in all_minutes:
        # Only include trading hours
        if minute.weekday() < 5:  # Monday to Friday
            hour = minute.hour
            min_val = minute.minute
            if (hour == 9 and min_val >= 15) or (9 < hour < 15) or (hour == 15 and min_val <= 30):
                trading_minutes.append(minute)
    
    # Reindex to include all trading minutes
    df = df.set_index('timestamp')
    full_index = pd.DatetimeIndex(trading_minutes)
    df = df.reindex(full_index)
    
    # Forward fill missing values (augment with previous minute's data)
    df = df.fillna(method='ffill')
    
    # Reset index
    df = df.reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    return df

def create_features(df):
    """Create rolling average and volume sum features"""
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 10-minute rolling average of close price (t-9 to t)
    df['rolling_avg_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    
    # 10-minute volume sum (t-10 to t) - Note: includes current minute
    df['volume_sum_10'] = df['volume'].rolling(window=10, min_periods=1).sum()
    
    # Target: 1 if price is higher 5 minutes later, 0 otherwise
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Remove rows where target cannot be computed (last 5 rows)
    df = df[:-5].copy()
    
    return df

def combine_stock_data(data_paths, version):
    """Combine multiple stock data files"""
    combined_data = []
    
    for path in data_paths:
        if os.path.exists(path):
            stock_name = Path(path).stem.split('__')[0]
            print(f"Processing {stock_name}")
            df = load_and_preprocess_data(path)
            df = create_features(df)
            df['stock'] = stock_name
            combined_data.append(df)
            print(f"  - Records: {len(df)}")
        else:
            print(f"Warning: {path} not found")
    
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        output_path = f'data/processed/processed_v{version}.csv'
        final_df.to_csv(output_path, index=False)
        print(f"Saved combined data to {output_path}")
        print(f"Total records: {len(final_df)}")
        print(f"Features: {list(final_df.columns)}")
        return final_df
    else:
        print("No data files found!")
        return None

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "0"
    
    if version == "0":
        # v0 data
        data_paths = [
            'data/v0/AARTIIND__EQ__NSE__NSE__MINUTE.csv',
            'data/v0/ABCAPITAL__EQ__NSE__NSE__MINUTE.csv'
        ]
    elif version == "1":
        # v1 data (includes v0 + v1)
        data_paths = [
            'data/v0/AARTIIND__EQ__NSE__NSE__MINUTE.csv',
            'data/v0/ABCAPITAL__EQ__NSE__NSE__MINUTE.csv',
            'data/v1/ABFRL__EQ__NSE__NSE__MINUTE.csv',
            'data/v1/ADANIENT__EQ__NSE__NSE__MINUTE.csv',
            'data/v1/ADANIGAS__EQ__NSE__NSE__MINUTE.csv'
        ]
    
    result = combine_stock_data(data_paths, version)
    if result is not None:
        print(f"Success! Data preprocessing completed for v{version}")
