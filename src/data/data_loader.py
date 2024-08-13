import torch
import numpy as np
from src.data.utils import load_data,stock_to_csv,extract_year_month
from torch.utils.data import TensorDataset, DataLoader

def create_data_loader(x,y, batch_size):
    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_prev_close_features(data,window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data['Close'].iloc[i - window_size:i])
        y.append(data['Close'].iloc[i])
    x = np.array(X)
    y = np.array(y)
    return x,y


def get_xy_for_ml_model_training(df,df_test):
    x, y = get_prev_close_features(df, window_size=13)

    train_sample = x.shape[0] - len(df_test)
    x_train, y_train = x[0:train_sample], y[0:train_sample]
    x_test, y_test = x[train_sample:], y[train_sample:]

    assert x_test.shape[0] == len(df_test)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return x_train,y_train,x_test, y_test

def stock_data_loader(stock_name,model_type):
    file_path = stock_to_csv[stock_name]
    print(f"loading data: {file_path}")
    df = load_data(file_path)
    df = extract_year_month(df)
    df_train, df_test = df[df['year'] < 2021], df[df['year'] >= 2021]
    print(f"data loaded.")
    print(f"total: {len(df)} \t train: {len(df_train)}, test: {len(df_test)}")
    if model_type in ['ma', 'ar', 'arma', 'arima']:
        return df_train, df_test
    elif model_type in ['linear_regression', 'svr', 'knn', 'random_forest']:
        x_train, y_train, x_test, y_test = get_xy_for_ml_model_training(df,df_test)
        return (x_train,y_train),(x_test, y_test,df_test)
    else:
        x_train, y_train, x_test, y_test = get_xy_for_ml_model_training(df, df_test)
        n_samples = -500
        x_train = x_train[n_samples:]
        y_train = y_train[n_samples:]
        return (x_train, y_train), (x_test, y_test, df_test)


