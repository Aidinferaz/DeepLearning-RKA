import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cyclical_encoder(df, col_name='Month', period=12):
    """
    Create sine and cosine transformations of a cyclical feature
    
    Parameters:
    df: DataFrame containing the column to encode
    col_name: Name of the column to encode
    period: The period of the cyclical feature (12 for months)
    
    Returns:
    DataFrame with two new columns: col_name_sin and col_name_cos
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert to numeric if needed
    if df_copy[col_name].dtype == 'object':
        # Map month names to numbers if needed
        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        df_copy[col_name] = df_copy[col_name].map(month_map)
    
    # Create cyclical features
    df_copy[f'{col_name}_sin'] = np.sin(2 * np.pi * df_copy[col_name]/period)
    df_copy[f'{col_name}_cos'] = np.cos(2 * np.pi * df_copy[col_name]/period)
    
    return df_copy

def xy_rnn_split(data, target_col=None, time_steps=1):
    """
    Split the data into X and y for RNN training.
    """
    if target_col is None:
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)
    else:
        # For univariate time series prediction (when target is also the feature)
        if len(data.columns) == 1:
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[target_col].values[i:i + time_steps])
                y.append(data[target_col].values[i + time_steps])
            return np.array(X), np.array(y).reshape(-1, 1)
        # For multivariate case
        else:
            feature = [col for col in data.columns if col != target_col]
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[feature].values[i:i + time_steps])
                y.append(data[target_col].values[i + time_steps])
            return np.array(X), np.array(y).reshape(-1, 1)

def create_sequences(data, sequence_length):
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]  # Ambil sequence selama 'sequence_length' bulan
        label = data[i+sequence_length]  # Nilai yang akan diprediksi (bulan berikutnya)

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)